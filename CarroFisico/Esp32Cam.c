/* ==============================================================================
 * PROYECTO 4WD SMART CAR ROBOT - MÓDULO ESP32-CAM (VISIÓN ARTIFICIAL)
 * Firmware ESP-IDF para streaming de video MJPEG de ultra baja latencia y telemetría
 * 
 * Hardware: AI-Thinker ESP32-CAM + Sensor OV2640 (2 Megapíxeles) + PSRAM
 * Red Wi-Fi (Inspirado en rakesh-i/ESP32-Autonomous-car wireless-setup):
 *   - Modo 100% Estación Pura (WIFI_MODE_STA) sin interferencia de AP local
 *   - Prioridad 1 (Conexión Directa a Laptop Hotspot): SSID "LAPTOP_ROBOT" / Pass "12345678"
 *     con IP Estática 192.168.137.48 (1 solo salto inalámbrico hacia la laptop)
 *   - Prioridad 2 (Router Local): Configurable vía .env o Kconfig (SSID / Pass)
 *     con IP Estática 192.168.100.48
 * Servidores:
 *   - Puerto 81: http://<IP_ESP32_CAM>:81/stream            (Video MJPEG asíncrono en Core 1)
 *   - Puerto 80: http://<IP_ESP32_CAM>/cam.jpg              (Captura de imagen estática)
 *   - Puerto 80: http://<IP_ESP32_CAM>/status               (Telemetría de cámara en JSON)
 *   - Puerto 80: http://<IP_ESP32_CAM>/flash?val=1          (Control de LED Flash frontal)
 *   - Puerto 80: http://<IP_ESP32_CAM>/control?var=..&val=.. (Ajuste de resolución/giro/calidad)
 * ============================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "lwip/inet.h"
#include "esp_http_server.h"
#include "esp_camera.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

static const char *TAG = "ESP32_CAM_MAIN";

/* ==============================================================================
 * CONFIGURACIÓN DE RED WI-FI: PRIORIDAD 1 (LAPTOP HOTSPOT) Y PRIORIDAD 2 (ROUTER)
 * ============================================================================== */
/* Opción A: Hotspot directo de la Laptop en Windows (1 solo salto Wi-Fi, mínima latencia) */
#ifndef LAPTOP_HOTSPOT_SSID
#define LAPTOP_HOTSPOT_SSID   "LAPTOP_ROBOT"
#endif
#ifndef LAPTOP_HOTSPOT_PASS
#define LAPTOP_HOTSPOT_PASS   "12345678"
#endif
#define LAPTOP_STATIC_IP      "192.168.137.48"
#define LAPTOP_GATEWAY_IP     "192.168.137.1"
#define LAPTOP_NETMASK        "255.255.255.0"

/* Opción B: Router local alternativo (con IP estática para evitar demoras de DHCP) */
#ifndef ROUTER_WIFI_SSID
#define ROUTER_WIFI_SSID      "TU_RED_WIFI"
#endif
#ifndef ROUTER_WIFI_PASS
#define ROUTER_WIFI_PASS      "TU_PASSWORD_WIFI"
#endif
#define ROUTER_STATIC_IP      "192.168.100.48"
#define ROUTER_GATEWAY_IP     "192.168.100.1"
#define ROUTER_NETMASK        "255.255.255.0"

static esp_netif_t *s_sta_netif = NULL;
static bool s_using_laptop_hotspot = false;
static int s_disconnect_count = 0;

/* ==============================================================================
 * PINES DE CONEXIÓN AI-THINKER ESP32-CAM (Sensor OV2640)
 * ============================================================================== */
#define CAM_PIN_PWDN    32
#define CAM_PIN_RESET   -1 // Software reset vía I2C SCCB
#define CAM_PIN_XCLK     0
#define CAM_PIN_SIOD    26 // SDA I2C
#define CAM_PIN_SIOC    27 // SCL I2C
#define CAM_PIN_Y9      35
#define CAM_PIN_Y8      34
#define CAM_PIN_Y7      39
#define CAM_PIN_Y6      36
#define CAM_PIN_Y5      21
#define CAM_PIN_Y4      19
#define CAM_PIN_Y3      18
#define CAM_PIN_Y2       5
#define CAM_PIN_VSYNC   25
#define CAM_PIN_HREF    23
#define CAM_PIN_PCLK    22

/* Pin del LED Flash de alta potencia (Blanco brillante) y LED rojo de estado */
#define FLASH_LED_PIN    4
#define STATUS_LED_PIN  33 // LED rojo integrado (activo a nivel bajo)

/* ==============================================================================
 * PARÁMETROS DEL STREAM MULTIPART MJPEG (HTTP RFC)
 * ============================================================================== */
#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_FRAME_HEADER = "\r\n--" PART_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

/* Estado global de telemetría y control asíncrono del stream */
static uint32_t s_frame_count = 0;
static float s_current_fps = 0.0f;
static int64_t s_last_fps_calc = 0;
static bool s_flash_enabled = false;
static int s_vflip = 0;
static int s_hmirror = 0;

/* ==============================================================================
 * CONFIGURACIÓN DE SOCKET TCP DE BAJA LATENCIA (TCP_NODELAY + TIMEOUT)
 * ============================================================================== */
static void configure_stream_socket(int fd) {
    if (fd < 0) return;

    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

/* ==============================================================================
 * TRANSMISIÓN DE VIDEO EN VIVO MJPEG DIRECTA (/stream en Puerto 81)
 * ============================================================================== */
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    char part_buf[128];

    int sockfd = httpd_req_to_sockfd(req);
    configure_stream_socket(sockfd);

    res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    httpd_resp_set_hdr(req, "Pragma", "no-cache");
    httpd_resp_set_hdr(req, "X-Framerate", "30");

    ESP_LOGI(TAG, "Stream MJPEG iniciado (socket %d)", sockfd);

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "Cuadro no disponible");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }

        s_frame_count++;
        int64_t now = esp_timer_get_time();
        if (now - s_last_fps_calc >= 1000000) {
            s_current_fps = (float)s_frame_count * 1000000.0f / (float)(now - s_last_fps_calc);
            s_frame_count = 0;
            s_last_fps_calc = now;
        }

        size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_FRAME_HEADER, (unsigned)fb->len);
        res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
        }
        esp_camera_fb_return(fb);

        if (res != ESP_OK) {
            ESP_LOGI(TAG, "Cliente cerro conexion del stream");
            break;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }

    return res;
}

/* ==============================================================================
 * MANEJADOR: Captura de Foto Individual (/cam.jpg en Puerto 80)
 * ============================================================================== */
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
    httpd_resp_set_hdr(req, "Connection", "close");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    return res;
}

/* ==============================================================================
 * MANEJADOR: Control de LED Flash (/flash?val=0|1)
 * ============================================================================== */
static esp_err_t flash_handler(httpd_req_t *req) {
    char buf[32];
    size_t buf_len = sizeof(buf);

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_set_type(req, "application/json");

    if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
        char param[16];
        if (httpd_query_key_value(buf, "val", param, sizeof(param)) == ESP_OK) {
            int val = atoi(param);
            s_flash_enabled = (val > 0);
            gpio_set_level(FLASH_LED_PIN, s_flash_enabled ? 1 : 0);
            ESP_LOGI(TAG, "Flash LED %s", s_flash_enabled ? "ENCENDIDO" : "APAGADO");
        }
    }

    char resp[64];
    snprintf(resp, sizeof(resp), "{\"flash\": %s}", s_flash_enabled ? "true" : "false");
    httpd_resp_sendstr(req, resp);
    return ESP_OK;
}

/* ==============================================================================
 * MANEJADOR: Ajustes Dinámicos de Sensor (/control?var=..&val=..)
 * ============================================================================== */
static esp_err_t control_handler(httpd_req_t *req) {
    char buf[64];
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_set_type(req, "application/json");

    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char var[24] = {0};
        char val_str[16] = {0};
        if (httpd_query_key_value(buf, "var", var, sizeof(var)) == ESP_OK &&
            httpd_query_key_value(buf, "val", val_str, sizeof(val_str)) == ESP_OK) {
            int val = atoi(val_str);
            sensor_t *s = esp_camera_sensor_get();
            if (s) {
                if (strcmp(var, "framesize") == 0) {
                    s->set_framesize(s, (framesize_t)val);
                    ESP_LOGI(TAG, "Resolucion cambiada a framesize=%d", val);
                } else if (strcmp(var, "quality") == 0) {
                    s->set_quality(s, val);
                    ESP_LOGI(TAG, "Calidad JPEG cambiada a %d", val);
                } else if (strcmp(var, "vflip") == 0) {
                    s_vflip = val ? 1 : 0;
                    s->set_vflip(s, s_vflip);
                    ESP_LOGI(TAG, "V-Flip=%d", s_vflip);
                } else if (strcmp(var, "hmirror") == 0) {
                    s_hmirror = val ? 1 : 0;
                    s->set_hmirror(s, s_hmirror);
                    ESP_LOGI(TAG, "H-Mirror=%d", s_hmirror);
                }
            }
        }
    }

    httpd_resp_sendstr(req, "{\"ok\":true}");
    return ESP_OK;
}

/* ==============================================================================
 * MANEJADOR: Telemetría de la Cámara en JSON (/status en Puerto 80)
 * ============================================================================== */
static esp_err_t status_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");

    wifi_ap_record_t ap_info;
    int rssi = 0;
    if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
        rssi = ap_info.rssi;
    }

    sensor_t *s = esp_camera_sensor_get();
    int frame_size = s ? s->status.framesize : -1;
    int quality = s ? s->status.quality : 24;

    char json_buf[300];
    snprintf(json_buf, sizeof(json_buf),
             "{"
             "\"device\":\"ESP32-CAM OV2640\","
             "\"mode\":\"%s\","
             "\"fps\":%.1f,"
             "\"framesize\":%d,"
             "\"quality\":%d,"
             "\"vflip\":%d,"
             "\"hmirror\":%d,"
             "\"flash\":%s,"
             "\"rssi\":%d,"
             "\"psram\":%s,"
             "\"free_heap\":%lu"
             "}",
             s_using_laptop_hotspot ? "LAPTOP_HOTSPOT" : "ROUTER_STA",
             s_current_fps,
             frame_size,
             quality,
             s_vflip,
             s_hmirror,
             s_flash_enabled ? "true" : "false",
             rssi,
             (heap_caps_get_free_size(MALLOC_CAP_SPIRAM) > 0) ? "true" : "false",
             (unsigned long)esp_get_free_heap_size());

    httpd_resp_sendstr(req, json_buf);
    return ESP_OK;
}

static esp_err_t favicon_handler(httpd_req_t *req) {
    httpd_resp_set_status(req, "204 No Content");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_send(req, NULL, 0);
    return ESP_OK;
}

/* ==============================================================================
 * INICIALIZACIÓN DE SERVIDORES HTTP (Puertos 80 y 81 en Core 1)
 * ============================================================================== */
void start_camera_webservers(void) {
    httpd_config_t config_80 = HTTPD_DEFAULT_CONFIG();
    config_80.server_port = 80;
    config_80.ctrl_port = 32768;
    config_80.core_id = 1;
    config_80.stack_size = 8192;
    config_80.max_open_sockets = 4;
    config_80.max_uri_handlers = 8;
    config_80.lru_purge_enable = true;
    config_80.send_wait_timeout = 2;
    config_80.recv_wait_timeout = 2;
    config_80.backlog_conn = 3;
    httpd_handle_t server_80 = NULL;

    if (httpd_start(&server_80, &config_80) == ESP_OK) {
        httpd_uri_t uri_capture = { .uri = "/cam.jpg",     .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
        httpd_uri_t uri_flash   = { .uri = "/flash",       .method = HTTP_GET, .handler = flash_handler,   .user_ctx = NULL };
        httpd_uri_t uri_control = { .uri = "/control",     .method = HTTP_GET, .handler = control_handler, .user_ctx = NULL };
        httpd_uri_t uri_status  = { .uri = "/status",      .method = HTTP_GET, .handler = status_handler,  .user_ctx = NULL };
        httpd_uri_t uri_favicon = { .uri = "/favicon.ico", .method = HTTP_GET, .handler = favicon_handler, .user_ctx = NULL };

        httpd_register_uri_handler(server_80, &uri_capture);
        httpd_register_uri_handler(server_80, &uri_flash);
        httpd_register_uri_handler(server_80, &uri_control);
        httpd_register_uri_handler(server_80, &uri_status);
        httpd_register_uri_handler(server_80, &uri_favicon);
        ESP_LOGI(TAG, "Servidor HTTP puerto 80 iniciado correctamente");
    } else {
        ESP_LOGE(TAG, "Fallo al iniciar servidor HTTP puerto 80");
    }

    httpd_config_t config_81 = HTTPD_DEFAULT_CONFIG();
    config_81.server_port = 81;
    config_81.ctrl_port = 32769;
    config_81.core_id = 1;
    config_81.stack_size = 8192;
    config_81.max_open_sockets = 3;
    config_81.lru_purge_enable = true;
    config_81.send_wait_timeout = 2;
    config_81.recv_wait_timeout = 2;
    config_81.backlog_conn = 2;
    httpd_handle_t server_81 = NULL;

    if (httpd_start(&server_81, &config_81) == ESP_OK) {
        httpd_uri_t uri_stream_81 = {
            .uri      = "/stream",
            .method   = HTTP_GET,
            .handler  = stream_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server_81, &uri_stream_81);
        ESP_LOGI(TAG, "Servidor Stream puerto 81 iniciado correctamente");
    } else {
        ESP_LOGE(TAG, "Fallo al iniciar servidor Stream puerto 81");
    }
}

/* ==============================================================================
 * CONFIGURACIÓN DE IP ESTÁTICA Y SELECCIÓN INTELIGENTE (LAPTOP HOTSPOT O ROUTER)
 * ============================================================================== */
static void apply_static_ip(const char *ip, const char *gw, const char *mask) {
    if (!s_sta_netif) return;
    esp_netif_dhcpc_stop(s_sta_netif);
    esp_netif_ip_info_t ip_info;
    memset(&ip_info, 0, sizeof(ip_info));
    ip_info.ip.addr = inet_addr(ip);
    ip_info.gw.addr = inet_addr(gw);
    ip_info.netmask.addr = inet_addr(mask);
    esp_netif_set_ip_info(s_sta_netif, &ip_info);
}

static void select_and_connect_wifi(void) {
    wifi_config_t wifi_config;
    memset(&wifi_config, 0, sizeof(wifi_config));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    /* Conexión directa y fija al Hotspot de Laptop ("LAPTOP_ROBOT") con IP 192.168.137.48 */
    s_using_laptop_hotspot = true;
    strncpy((char *)wifi_config.sta.ssid, LAPTOP_HOTSPOT_SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char *)wifi_config.sta.password, LAPTOP_HOTSPOT_PASS, sizeof(wifi_config.sta.password));
    apply_static_ip(LAPTOP_STATIC_IP, LAPTOP_GATEWAY_IP, LAPTOP_NETMASK);

    ESP_LOGI(TAG, "Conectando directo al Hotspot de Laptop: %s (IP estatica: %s)...", LAPTOP_HOTSPOT_SSID, LAPTOP_STATIC_IP);
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    esp_wifi_connect();
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_disconnect_count++;
        ESP_LOGW(TAG, "Wi-Fi desconectado (intento %d), reconectando...", s_disconnect_count);
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        s_disconnect_count = 0;

        /* Encender LED rojo integrado (activo bajo) brevemente para indicar conexión exitosa */
        gpio_set_level(STATUS_LED_PIN, 0);

        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "  ESP32-CAM CONECTADO (%s)", s_using_laptop_hotspot ? "HOTSPOT LAPTOP DIRECTO" : "ROUTER LOCAL");
        ESP_LOGI(TAG, "  Direccion IP: " IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "  Video Stream: http://" IPSTR ":81/stream", IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "==========================================================");
    }
}

/* ==============================================================================
 * INICIALIZACIÓN DE LA CÁMARA OV2640 (Reloj 20 MHz nativo como en CameraWebServer)
 * ============================================================================== */
esp_err_t init_camera(void) {
    camera_config_t config;
    memset(&config, 0, sizeof(camera_config_t));

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = CAM_PIN_Y2;
    config.pin_d1       = CAM_PIN_Y3;
    config.pin_d2       = CAM_PIN_Y4;
    config.pin_d3       = CAM_PIN_Y5;
    config.pin_d4       = CAM_PIN_Y6;
    config.pin_d5       = CAM_PIN_Y7;
    config.pin_d6       = CAM_PIN_Y8;
    config.pin_d7       = CAM_PIN_Y9;
    config.pin_xclk     = CAM_PIN_XCLK;
    config.pin_pclk     = CAM_PIN_PCLK;
    config.pin_vsync    = CAM_PIN_VSYNC;
    config.pin_href     = CAM_PIN_HREF;
    config.pin_sccb_sda = CAM_PIN_SIOD;
    config.pin_sccb_scl = CAM_PIN_SIOC;
    config.pin_pwdn     = CAM_PIN_PWDN;
    config.pin_reset    = CAM_PIN_RESET;

    /* 10 MHz para máxima estabilidad de reloj XCLK y DMA sin pérdidas de paquetes */
    config.xclk_freq_hz = 10000000;

    /* Resolución QVGA (320x240) con calidad 24 (~3.5-5 KB por cuadro = 0 congestión Wi-Fi) */
    config.frame_size   = FRAMESIZE_QVGA;
    config.pixel_format = PIXFORMAT_JPEG;
    config.jpeg_quality = 24;
    config.fb_count     = 2;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Fallo al inicializar camara OV2640 (Error: 0x%x)", err);
        return err;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, 0);
        s->set_hmirror(s, 0);
        s->set_brightness(s, 1);
        s->set_contrast(s, 1);
        s->set_saturation(s, 0);
    }

    return ESP_OK;
}

void wifi_init_sta(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_sta_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    cfg.ampdu_rx_enable = 0;
    cfg.ampdu_tx_enable = 0;
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_log_level_set("wifi", ESP_LOG_WARN);
    esp_log_level_set("cam_hal", ESP_LOG_WARN);

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    /* Modo Estación Puro (WIFI_MODE_STA) */
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(48)); // 12 dBm para estabilidad electrica con alimentación USB

    select_and_connect_wifi();
}

void init_leds(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << FLASH_LED_PIN) | (1ULL << STATUS_LED_PIN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&io_conf);

    gpio_set_level(FLASH_LED_PIN, 0);
    gpio_set_level(STATUS_LED_PIN, 1);
}

void app_main(void) {
    WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0); // Desactivar brownout detector para evitar reseteos por consumo transitorio de RF

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    init_leds();

    if (init_camera() != ESP_OK) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }

    wifi_init_sta();
    start_camera_webservers();
}
