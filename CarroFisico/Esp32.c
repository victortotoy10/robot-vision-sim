/* ==============================================================================
 * PROYECTO 4WD SMART CAR ROBOT - MÓDULO ESP32 (CONTROL MOTRIZ Y CINEMÁTICA)
 * Firmware ESP-IDF para control de tracción diferencial skid-steering (Puente H L298N)
 * 
 * Hardware:
 *   - ESP32 DevKit V1 (38 pines / 30 pines)
 *   - Módulo Controlador Puente H Doble L298N
 *   - 4x Motores DC con Reductora TT (1:48)
 * 
 * Asignación de Pines Seguros (L298N):
 *   - Motores Lado Derecho:  ENA = GPIO 32 (PWM), IN1 = GPIO 33, IN2 = GPIO 22
 *   - Motores Lado Izquierdo: IN3 = GPIO 27, IN4 = GPIO 26, ENB = GPIO 25 (PWM)
 * 
 * Red Wi-Fi de Baja Latencia (Basado en rakesh-i/ESP32-Autonomous-car wireless-setup):
 *   - Modo 100% Estación Pura (WIFI_MODE_STA): Desactiva el SoftAP simultáneo para
 *     eliminar la interferencia de radiofrecuencia de 2.4 GHz sobre la ESP32-CAM en el chasis.
 *   - Prioridad 1 (Hotspot Directo de Laptop): SSID "LAPTOP_ROBOT" / Pass "12345678"
 *     con IP Estática 192.168.137.47 (1 solo salto inalámbrico directo a la laptop).
 *   - Prioridad 2 (Router Local): Configurable vía .env o Kconfig (SSID / Pass)
 *     con IP Estática 192.168.100.47.
 * ============================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_log.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "lwip/inet.h"
#include "esp_http_server.h"

static const char *TAG = "ESP32_MOTOR_CORE";

/* ==============================================================================
 * CONFIGURACIÓN DE PINES DEL CONTROLADOR DE MOTORES L298N
 * ============================================================================== */
#define ENA_PIN       32
#define IN1_PIN       33
#define IN2_PIN       22

#define IN3_PIN       27
#define IN4_PIN       26
#define ENB_PIN       25

#define LEDC_TIMER_NUM          LEDC_TIMER_0
#define LEDC_SPEED_MODE         LEDC_LOW_SPEED_MODE
#define LEDC_DUTY_RESOLUTION    LEDC_TIMER_8_BIT
#define LEDC_PWM_FREQUENCY      1000
#define LEDC_CHANNEL_RIGHT      LEDC_CHANNEL_0
#define LEDC_CHANNEL_LEFT       LEDC_CHANNEL_1

/* ==============================================================================
 * CONFIGURACIÓN WI-FI: PRIORIDAD 1 (LAPTOP HOTSPOT) Y PRIORIDAD 2 (ROUTER)
 * ============================================================================== */
#ifndef LAPTOP_HOTSPOT_SSID
#define LAPTOP_HOTSPOT_SSID     "LAPTOP_ROBOT"
#endif
#ifndef LAPTOP_HOTSPOT_PASS
#define LAPTOP_HOTSPOT_PASS     "12345678"
#endif
#define LAPTOP_STATIC_IP        "192.168.137.47"
#define LAPTOP_GATEWAY_IP       "192.168.137.1"
#define LAPTOP_NETMASK          "255.255.255.0"

#ifndef ROUTER_WIFI_SSID
#define ROUTER_WIFI_SSID        "TU_RED_WIFI"
#endif
#ifndef ROUTER_WIFI_PASS
#define ROUTER_WIFI_PASS        "TU_PASSWORD_WIFI"
#endif
#define ROUTER_STATIC_IP        "192.168.100.47"
#define ROUTER_GATEWAY_IP       "192.168.100.1"
#define ROUTER_NETMASK          "255.255.255.0"
static esp_netif_t *s_sta_netif = NULL;
static bool s_using_laptop_hotspot = false;

/* ==============================================================================
 * ESTADO Y VARIABLES DE CONTROL CINEMÁTICO
 * ============================================================================== */
typedef enum {
    MOTOR_STATE_STOP = 0,
    MOTOR_STATE_FWD,
    MOTOR_STATE_REV,
    MOTOR_STATE_LEFT,
    MOTOR_STATE_RIGHT,
    MOTOR_STATE_CUSTOM
} motor_state_t;

static motor_state_t s_current_state = MOTOR_STATE_STOP;
static int s_base_speed = 220;
static int s_turn_speed = 190;
static int s_current_speed_left  = 0;
static int s_current_speed_right = 0;

static int64_t s_last_command_time = 0;
#define WATCHDOG_TIMEOUT_US (1500000)

/* ==============================================================================
 * CONTROL HARDWARE DE MOTORES (GPIO + PWM LEDC)
 * ============================================================================== */
void init_motors(void) {
    gpio_config_t io_conf = {
        .intr_type    = GPIO_INTR_DISABLE,
        .mode         = GPIO_MODE_OUTPUT,
        .pin_bit_mask = (1ULL << IN1_PIN) | (1ULL << IN2_PIN) |
                        (1ULL << IN3_PIN) | (1ULL << IN4_PIN),
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .pull_up_en   = GPIO_PULLUP_DISABLE
    };
    gpio_config(&io_conf);

    ledc_timer_config_t timer_conf = {
        .speed_mode      = LEDC_SPEED_MODE,
        .timer_num       = LEDC_TIMER_NUM,
        .duty_resolution = LEDC_DUTY_RESOLUTION,
        .freq_hz         = LEDC_PWM_FREQUENCY,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ledc_timer_config(&timer_conf);

    ledc_channel_config_t ch_ena = {
        .speed_mode = LEDC_SPEED_MODE,
        .channel    = LEDC_CHANNEL_RIGHT,
        .timer_sel  = LEDC_TIMER_NUM,
        .intr_type  = LEDC_INTR_DISABLE,
        .gpio_num   = ENA_PIN,
        .duty       = 0,
        .hpoint     = 0
    };
    ledc_channel_config(&ch_ena);

    ledc_channel_config_t ch_enb = {
        .speed_mode = LEDC_SPEED_MODE,
        .channel    = LEDC_CHANNEL_LEFT,
        .timer_sel  = LEDC_TIMER_NUM,
        .intr_type  = LEDC_INTR_DISABLE,
        .gpio_num   = ENB_PIN,
        .duty       = 0,
        .hpoint     = 0
    };
    ledc_channel_config(&ch_enb);

    gpio_set_level(IN1_PIN, 0);
    gpio_set_level(IN2_PIN, 0);
    gpio_set_level(IN3_PIN, 0);
    gpio_set_level(IN4_PIN, 0);
    ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_RIGHT, 0);
    ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_RIGHT);
    ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_LEFT, 0);
    ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_LEFT);
}

void apply_motor_raw(int in1, int in2, int in3, int in4, int speed_right, int speed_left) {
    gpio_set_level(IN1_PIN, in1);
    gpio_set_level(IN2_PIN, in2);
    gpio_set_level(IN3_PIN, in3);
    gpio_set_level(IN4_PIN, in4);

    if (speed_right < 0) speed_right = 0;
    if (speed_right > 255) speed_right = 255;
    if (speed_left < 0) speed_left = 0;
    if (speed_left > 255) speed_left = 255;

    ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_RIGHT, speed_right);
    ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_RIGHT);

    ledc_set_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_LEFT, speed_left);
    ledc_update_duty(LEDC_SPEED_MODE, LEDC_CHANNEL_LEFT);

    s_current_speed_right = speed_right;
    s_current_speed_left  = speed_left;
    s_last_command_time   = esp_timer_get_time();
}

void set_robot_motion(motor_state_t state) {
    s_current_state = state;
    switch (state) {
        case MOTOR_STATE_FWD:
            apply_motor_raw(1, 0, 1, 0, s_base_speed, s_base_speed);
            break;
        case MOTOR_STATE_REV:
            apply_motor_raw(0, 1, 0, 1, s_base_speed, s_base_speed);
            break;
        case MOTOR_STATE_LEFT:
            apply_motor_raw(1, 0, 0, 1, s_turn_speed, s_turn_speed);
            break;
        case MOTOR_STATE_RIGHT:
            apply_motor_raw(0, 1, 1, 0, s_turn_speed, s_turn_speed);
            break;
        case MOTOR_STATE_STOP:
        default:
            apply_motor_raw(0, 0, 0, 0, 0, 0);
            break;
    }
}

void set_differential_drive(int left_pwm, int right_pwm) {
    s_current_state = MOTOR_STATE_CUSTOM;
    int in1 = 0, in2 = 0, in3 = 0, in4 = 0;

    if (right_pwm > 0) {
        in1 = 1; in2 = 0;
    } else if (right_pwm < 0) {
        in1 = 0; in2 = 1;
        right_pwm = -right_pwm;
    }

    if (left_pwm > 0) {
        in3 = 1; in4 = 0;
    } else if (left_pwm < 0) {
        in3 = 0; in4 = 1;
        left_pwm = -left_pwm;
    }

    apply_motor_raw(in1, in2, in3, in4, right_pwm, left_pwm);
}

static void watchdog_task(void *pvParameters) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(100));
        if (s_current_state != MOTOR_STATE_STOP) {
            int64_t now = esp_timer_get_time();
            if (now - s_last_command_time > WATCHDOG_TIMEOUT_US) {
                set_robot_motion(MOTOR_STATE_STOP);
            }
        }
    }
}

/* ==============================================================================
 * INTERFAZ WEB EMBEBIDA MEJORADA (HUD CONTROL REMOTO SIN POLLING INTRUSIVO)
 * ============================================================================== */
static const char HTML_INDEX[] =
"<!DOCTYPE html><html lang='es'><head><meta charset='UTF-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no'>"
"<title>4WD Smart Car Remote</title>"
"<style>"
"*{box-sizing:border-box;margin:0;padding:0;user-select:none;-webkit-user-select:none;}"
"body{background:#070b14;color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;display:flex;flex-direction:column;align-items:center;min-height:100vh;padding:10px;}"
".app-shell{width:100%;max-width:400px;display:flex;flex-direction:column;gap:10px;}"
".top-bar{display:flex;justify-content:space-between;align-items:center;background:#111827;border:1px solid #1f2937;border-radius:12px;padding:8px 12px;box-shadow:0 4px 14px rgba(0,0,0,0.4);}"
".brand{display:flex;align-items:center;gap:7px;font-weight:800;font-size:14px;letter-spacing:0.6px;color:#38bdf8;}"
".dot{width:9px;height:9px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444;transition:all 0.2s;}"
".dot.on{background:#22c55e;box-shadow:0 0 8px #22c55e;}"
".hud-badges{display:flex;gap:6px;font-size:11px;font-family:monospace;align-items:center;}"
".badge{background:#1e293b;border:1px solid #334155;padding:3px 7px;border-radius:6px;color:#cbd5e1;cursor:pointer;}"
".badge b{color:#38bdf8;}"
".stream-card{background:#111827;border:1px solid #1f2937;border-radius:14px;overflow:hidden;box-shadow:0 10px 25px rgba(0,0,0,0.55);}"
".viewport{position:relative;width:100%;aspect-ratio:4/3;background:#030712;display:flex;align-items:center;justify-content:center;overflow:hidden;}"
".viewport img{width:100%;height:100%;object-fit:cover;display:block;}"
".cam-overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(3,7,18,0.78);color:#94a3b8;font-size:12px;gap:6px;pointer-events:none;transition:opacity 0.2s;}"
".cam-overlay.hidden{opacity:0;}"
".stream-bar{display:flex;padding:7px 8px;gap:6px;background:#0f172a;border-top:1px solid #1e293b;align-items:center;}"
".stream-bar input{flex:1;min-width:0;background:#1e293b;border:1px solid #334155;color:#f8fafc;border-radius:7px;padding:6px 8px;font-size:12px;font-family:monospace;outline:none;}"
".stream-bar select{background:#1e293b;border:1px solid #334155;color:#38bdf8;border-radius:7px;padding:6px 5px;font-size:11px;font-weight:600;outline:none;}"
".btn-conn{background:#38bdf8;color:#020617;border:none;border-radius:7px;padding:6px 11px;font-weight:700;font-size:12px;cursor:pointer;}"
".cam-tools{display:grid;grid-template-columns:1.3fr 1fr auto auto auto;gap:5px;padding:7px 8px;background:#0b1120;border-top:1px solid #1e293b;align-items:center;}"
".tool-btn{background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:7px;padding:6px 8px;font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:4px;transition:all 0.15s;}"
".tool-btn:active{transform:scale(0.95);}"
".tool-btn.flash-on{background:#f59e0b;color:#020617;border-color:#fbbf24;box-shadow:0 0 12px rgba(245,158,11,0.5);}"
".tool-btn.active-opt{border-color:#38bdf8;color:#38bdf8;}"
".cam-tools select{background:#1e293b;border:1px solid #334155;color:#e2e8f0;border-radius:7px;padding:5px 4px;font-size:11px;outline:none;}"
".control-card{background:#111827;border:1px solid #1f2937;border-radius:14px;padding:12px;display:flex;flex-direction:column;align-items:center;gap:10px;}"
".dpad{display:grid;grid-template-columns:repeat(3,82px);grid-template-rows:repeat(3,72px);gap:8px;justify-content:center;}"
".dpad button{width:82px;height:72px;border-radius:14px;border:1px solid #334155;background:#1e293b;color:#38bdf8;font-size:22px;font-weight:800;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;cursor:pointer;box-shadow:0 4px 10px rgba(0,0,0,0.4);touch-action:none;transition:all 0.06s;}"
".dpad button span{font-size:9px;color:#94a3b8;font-weight:600;letter-spacing:0.5px;}"
".dpad button.pressed,.dpad button:active{background:#38bdf8;color:#020617;border-color:#7dd3fc;transform:scale(0.94);box-shadow:0 0 15px rgba(56,189,248,0.45);}"
".dpad button.pressed span,.dpad button:active span{color:#020617;}"
".dpad button.btn-stop{background:#dc2626;color:#fff;border-color:#ef4444;}"
".dpad button.btn-stop span{color:#fecaca;}"
".dpad button.btn-stop.pressed,.dpad button.btn-stop:active{background:#b91c1c;color:#fff;}"
".speed-box{width:100%;background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:9px 12px;display:flex;flex-direction:column;gap:7px;}"
".speed-hdr{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#94a3b8;}"
".speed-presets{display:flex;gap:5px;}"
".speed-presets button{background:#1e293b;border:1px solid #334155;color:#cbd5e1;border-radius:5px;padding:3px 7px;font-size:10px;font-weight:700;cursor:pointer;}"
".speed-presets button:hover{border-color:#38bdf8;color:#38bdf8;}"
"input[type=range]{width:100%;accent-color:#38bdf8;cursor:pointer;}"
"</style><script>"
"let isFlash=false,isVflip=false,isHmirror=false,isConnected=false,pollTimer=null,retryTimer=null,holdTimer=null,activeCmd=null;"
"function getCleanIp(){let r=(document.getElementById('camIp').value||'').trim();return r.replace(/^https?:\\/\\//i,'').split(':')[0].split('/')[0];}"
"function setMotorBadge(c){const el=document.getElementById('stMotor');if(el)el.innerText=c.toUpperCase();}"
"function sendCmd(c){setMotorBadge(c);fetch('/'+c,{method:'GET',mode:'no-cors',cache:'no-store'}).catch(()=>{});}"
"function startHold(c,btnId){"
"if(activeCmd===c)return;"
"stopHold(false);"
"activeCmd=c;"
"if(btnId){const b=document.getElementById(btnId);if(b)b.classList.add('pressed');}"
"sendCmd(c);"
"holdTimer=setInterval(()=>sendCmd(c),450);"
"}"
"function stopHold(sendStop=true){"
"if(holdTimer){clearInterval(holdTimer);holdTimer=null;}"
"activeCmd=null;"
"['btnF','btnB','btnL','btnR','btnS'].forEach(id=>{const b=document.getElementById(id);if(b)b.classList.remove('pressed');});"
"if(sendStop)sendCmd('stop');"
"}"
"function updateSpeed(v){document.getElementById('spRange').value=v;document.getElementById('spVal').innerText=v;fetch('/speed?val='+v,{mode:'no-cors',cache:'no-store'}).catch(()=>{});}"
"function attach(id,c){const b=document.getElementById(id);if(!b)return;"
"b.addEventListener('pointerdown',e=>{e.preventDefault();startHold(c,id);});"
"b.addEventListener('pointerup',e=>{e.preventDefault();stopHold(true);});"
"b.addEventListener('pointercancel',e=>{e.preventDefault();stopHold(true);});"
"b.addEventListener('pointerleave',e=>{if(activeCmd===c){e.preventDefault();stopHold(true);}});}"
"function toggleFlash(){"
"const ip=getCleanIp();if(!ip)return;"
"isFlash=!isFlash;const val=isFlash?1:0;"
"fetch('http://'+ip+'/flash?val='+val,{mode:'no-cors',cache:'no-store'}).catch(()=>{});"
"const bf=document.getElementById('btnFlash');const tf=document.getElementById('flashTxt');"
"if(isFlash){bf.classList.add('flash-on');tf.innerText='ON';}else{bf.classList.remove('flash-on');tf.innerText='OFF';}"
"}"
"function camCtrl(v,val){const ip=getCleanIp();if(!ip)return;fetch('http://'+ip+'/control?var='+v+'&val='+val,{mode:'no-cors',cache:'no-store'}).catch(()=>{});}"
"function toggleFlip(){isVflip=!isVflip;document.getElementById('btnVflip').classList.toggle('active-opt',isVflip);camCtrl('vflip',isVflip?1:0);}"
"function toggleMirror(){isHmirror=!isHmirror;document.getElementById('btnHmir').classList.toggle('active-opt',isHmirror);camCtrl('hmirror',isHmirror?1:0);}"
"function showOverlay(msg){const ov=document.getElementById('camOv');if(!ov)return;if(msg){ov.innerText=msg;ov.classList.remove('hidden');}else{ov.classList.add('hidden');}}"
"function setDot(ok){const d=document.getElementById('camDot');if(d)d.classList.toggle('on',ok);}"
"function stopPolling(){if(pollTimer){clearTimeout(pollTimer);pollTimer=null;}if(retryTimer){clearTimeout(retryTimer);retryTimer=null;}}"
"function runPolling(){"
"if(!isConnected)return;const ip=getCleanIp();if(!ip)return;"
"const tmp=new Image();"
"tmp.onload=()=>{if(!isConnected)return;document.getElementById('feed').src=tmp.src;showOverlay('');setDot(true);pollTimer=setTimeout(runPolling,55);};"
"tmp.onerror=()=>{if(!isConnected)return;setDot(false);pollTimer=setTimeout(runPolling,250);};"
"tmp.src='http://'+ip+'/cam.jpg?t='+Date.now();"
"}"
"function checkCamTelemetryOnce(){"
"const ip=getCleanIp();if(!ip)return;"
"fetch('http://'+ip+'/status',{cache:'no-store'}).then(r=>r.json()).then(d=>{"
"if(d.rssi!==undefined)document.getElementById('stRssi').innerText=d.rssi+' dBm';"
"}).catch(()=>{});"
"}"
"function setStream(){"
"const ip=getCleanIp();if(!ip)return;"
"localStorage.setItem('esp32_cam_ip',ip);"
"stopPolling();"
"const mode=document.getElementById('selMode').value;"
"const img=document.getElementById('feed');"
"isConnected=true;"
"showOverlay('Conectando con ESP32-CAM ('+ip+')...');"
"if(mode==='snap'){runPolling();}else{"
"img.src='';"
"retryTimer=setTimeout(()=>{if(isConnected)img.src='http://'+ip+':81/stream?t='+Date.now();},80);"
"}"
"const bs=document.getElementById('btnStream');"
"bs.innerText='Cortar';bs.style.background='#ef4444';bs.style.color='#fff';"
"}"
"function onStreamLoad(){showOverlay('');setDot(true);}"
"function onStreamError(){"
"if(!isConnected)return;"
"setDot(false);"
"const mode=document.getElementById('selMode').value;"
"if(mode==='mjpeg'){"
"showOverlay('Reconectando stream MJPEG...');"
"if(retryTimer)clearTimeout(retryTimer);"
"retryTimer=setTimeout(()=>{if(isConnected)setStream();},900);"
"}"
"}"
"function toggleStream(){"
"if(isConnected){"
"stopPolling();isConnected=false;setDot(false);document.getElementById('feed').src='';"
"showOverlay('Stream detenido');"
"const bs=document.getElementById('btnStream');bs.innerText='Conectar';bs.style.background='#38bdf8';bs.style.color='#020617';"
"}else{setStream();}"
"}"
"function reloadStream(){if(isConnected){setStream();}}"
"window.onload=()=>{attach('btnF','fwd');attach('btnB','rev');attach('btnL','left');attach('btnR','right');"
"document.getElementById('btnS').onclick=()=>stopHold(true);"
"let defIp='192.168.100.48';"
"if(location.hostname.startsWith('192.168.137.'))defIp='192.168.137.48';"
"const saved=localStorage.getItem('esp32_cam_ip')||defIp;"
"document.getElementById('camIp').value=(location.hostname.startsWith('192.168.137.'))?'192.168.137.48':saved;"
"checkCamTelemetryOnce();"
"setStream();"
"window.addEventListener('keydown',e=>{"
"if(document.activeElement&&document.activeElement.tagName==='INPUT')return;"
"if(e.repeat)return;"
"const k=e.key.toLowerCase();"
"if(k==='w'||e.key==='ArrowUp'){e.preventDefault();startHold('fwd','btnF');}"
"else if(k==='s'||e.key==='ArrowDown'){e.preventDefault();startHold('rev','btnB');}"
"else if(k==='a'||e.key==='ArrowLeft'){e.preventDefault();startHold('left','btnL');}"
"else if(k==='d'||e.key==='ArrowRight'){e.preventDefault();startHold('right','btnR');}"
"else if(e.key===' '||e.key==='Escape'){e.preventDefault();stopHold(true);}"
"else if(k==='f'){e.preventDefault();toggleFlash();}"
"});"
"window.addEventListener('keyup',e=>{"
"if(document.activeElement&&document.activeElement.tagName==='INPUT')return;"
"const k=e.key.toLowerCase();"
"if(['w','s','a','d','arrowup','arrowdown','arrowleft','arrowright'].includes(k)){e.preventDefault();stopHold(true);}"
"});"
"};"
"</script></head><body>"
"<div class='app-shell'>"
"<div class='top-bar'>"
"<div class='brand'><div id='camDot' class='dot'></div>4WD SMART CAR</div>"
"<div class='hud-badges'>"
"<div class='badge' id='stRssi' onclick='checkCamTelemetryOnce()' title='Clic para medir señal Wi-Fi'>Wi-Fi OK</div>"
"<div class='badge'><b id='stMotor'>STOP</b></div>"
"</div>"
"</div>"
"<div class='stream-card'>"
"<div class='viewport'>"
"<img id='feed' alt='' src='' onload='onStreamLoad()' onerror='onStreamError()'>"
"<div id='camOv' class='cam-overlay'>Iniciando cámara...</div>"
"</div>"
"<div class='stream-bar'>"
"<input type='text' id='camIp' placeholder='IP Cam (192.168.100.48)'>"
"<select id='selMode' onchange='if(isConnected)setStream();'>"
"<option value='mjpeg'>MJPEG :81</option>"
"<option value='snap'>Snap :80</option>"
"</select>"
"<button id='btnStream' class='btn-conn' onclick='toggleStream()'>Conectar</button>"
"</div>"
"<div class='cam-tools'>"
"<button id='btnFlash' class='tool-btn' onclick='toggleFlash()' title='Tecla F'>💡 Luz: <span id='flashTxt'>OFF</span></button>"
"<select id='selRes' onchange='camCtrl(\"framesize\",this.value)' title='Resolución de Cámara'>"
"<option value='5' selected>QVGA 320x240</option>"
"<option value='3'>QQVGA 160x120 (Rápido)</option>"
"<option value='6'>CIF 400x296</option>"
"</select>"
"<button id='btnVflip' class='tool-btn' onclick='toggleFlip()' title='Voltear Vertical'>↕️</button>"
"<button id='btnHmir' class='tool-btn' onclick='toggleMirror()' title='Espejo Horizontal'>↔️</button>"
"<button class='tool-btn' onclick='reloadStream()' title='Reconectar Stream'>🔄</button>"
"</div>"
"</div>"
"<div class='control-card'>"
"<div class='dpad'>"
"<div></div><button id='btnF'>&#9650;<span>W / AVANZAR</span></button><div></div>"
"<button id='btnL'>&#9664;<span>A / IZQ</span></button>"
"<button id='btnS' class='btn-stop'>&#9632;<span>STOP</span></button>"
"<button id='btnR'>&#9654;<span>D / DER</span></button>"
"<div></div><button id='btnB'>&#9660;<span>S / ATRÁS</span></button><div></div>"
"</div>"
"<div class='speed-box'>"
"<div class='speed-hdr'>"
"<span>Potencia PWM: <b id='spVal' style='color:#38bdf8'>220</b> / 255</span>"
"<div class='speed-presets'>"
"<button onclick='updateSpeed(150)'>Suave</button>"
"<button onclick='updateSpeed(200)'>Normal</button>"
"<button onclick='updateSpeed(255)'>Máx</button>"
"</div>"
"</div>"
"<input id='spRange' type='range' min='100' max='255' value='220' oninput='updateSpeed(this.value)'>"
"</div>"
"</div>"
"</div>"
"</body></html>";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");
    return httpd_resp_send(req, HTML_INDEX, strlen(HTML_INDEX));
}

static esp_err_t cmd_handler(httpd_req_t *req) {
    const char *uri = req->uri;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_set_type(req, "text/plain");

    if (strcmp(uri, "/fwd") == 0) {
        set_robot_motion(MOTOR_STATE_FWD);
    } else if (strcmp(uri, "/rev") == 0) {
        set_robot_motion(MOTOR_STATE_REV);
    } else if (strcmp(uri, "/left") == 0) {
        set_robot_motion(MOTOR_STATE_LEFT);
    } else if (strcmp(uri, "/right") == 0) {
        set_robot_motion(MOTOR_STATE_RIGHT);
    } else if (strcmp(uri, "/stop") == 0) {
        set_robot_motion(MOTOR_STATE_STOP);
    }

    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t drive_handler(httpd_req_t *req) {
    char buf[64];
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");
    httpd_resp_set_type(req, "application/json");

    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char left_str[16] = {0};
        char right_str[16] = {0};
        httpd_query_key_value(buf, "l", left_str, sizeof(left_str));
        httpd_query_key_value(buf, "r", right_str, sizeof(right_str));

        if (strlen(left_str) > 0 && strlen(right_str) > 0) {
            int left_val = atoi(left_str);
            int right_val = atoi(right_str);
            set_differential_drive(left_val, right_val);
            return httpd_resp_sendstr(req, "{\"status\":\"ok\"}");
        }
    }
    return httpd_resp_send_404(req);
}

static esp_err_t speed_handler(httpd_req_t *req) {
    char buf[32];
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");

    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char val_str[16];
        if (httpd_query_key_value(buf, "val", val_str, sizeof(val_str)) == ESP_OK) {
            int val = atoi(val_str);
            if (val >= 60 && val <= 255) {
                s_base_speed = val;
                s_turn_speed = (int)(val * 0.85);
            }
        }
    }
    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t status_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_set_hdr(req, "Connection", "close");

    char resp[200];
    const char *state_names[] = {"STOP", "FWD", "REV", "LEFT", "RIGHT", "CUSTOM"};
    snprintf(resp, sizeof(resp),
             "{\"state\":\"%s\",\"mode\":\"%s\",\"speed_l\":%d,\"speed_r\":%d,\"base_speed\":%d,\"uptime_ms\":%lld}",
             state_names[s_current_state],
             s_using_laptop_hotspot ? "LAPTOP_HOTSPOT" : "ROUTER_STA",
             s_current_speed_left,
             s_current_speed_right,
             s_base_speed,
             (long long)(esp_timer_get_time() / 1000));

    return httpd_resp_sendstr(req, resp);
}

void start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    config.max_uri_handlers = 12;
    config.lru_purge_enable = true;
    config.max_open_sockets = 6;
    config.send_wait_timeout = 2;
    config.recv_wait_timeout = 2;
    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t uri_index = { .uri = "/",       .method = HTTP_GET, .handler = index_handler,  .user_ctx = NULL };
        httpd_uri_t uri_fwd   = { .uri = "/fwd",    .method = HTTP_GET, .handler = cmd_handler,    .user_ctx = NULL };
        httpd_uri_t uri_rev   = { .uri = "/rev",    .method = HTTP_GET, .handler = cmd_handler,    .user_ctx = NULL };
        httpd_uri_t uri_left  = { .uri = "/left",   .method = HTTP_GET, .handler = cmd_handler,    .user_ctx = NULL };
        httpd_uri_t uri_right = { .uri = "/right",  .method = HTTP_GET, .handler = cmd_handler,    .user_ctx = NULL };
        httpd_uri_t uri_stop  = { .uri = "/stop",   .method = HTTP_GET, .handler = cmd_handler,    .user_ctx = NULL };
        httpd_uri_t uri_drive = { .uri = "/drive",  .method = HTTP_GET, .handler = drive_handler,  .user_ctx = NULL };
        httpd_uri_t uri_speed = { .uri = "/speed",  .method = HTTP_GET, .handler = speed_handler,  .user_ctx = NULL };
        httpd_uri_t uri_stat  = { .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL };

        httpd_register_uri_handler(server, &uri_index);
        httpd_register_uri_handler(server, &uri_fwd);
        httpd_register_uri_handler(server, &uri_rev);
        httpd_register_uri_handler(server, &uri_left);
        httpd_register_uri_handler(server, &uri_right);
        httpd_register_uri_handler(server, &uri_stop);
        httpd_register_uri_handler(server, &uri_drive);
        httpd_register_uri_handler(server, &uri_speed);
        httpd_register_uri_handler(server, &uri_stat);
    }
}

/* ==============================================================================
 * CONFIGURACIÓN WI-FI ESTACIÓN PURA (SIN INTERFERENCIA DE AP EN EL CHASIS)
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
    wifi_scan_config_t scan_cfg = {
        .ssid = (uint8_t *)LAPTOP_HOTSPOT_SSID,
        .bssid = NULL,
        .channel = 0,
        .show_hidden = true
    };

    bool found_laptop = false;
    if (esp_wifi_scan_start(&scan_cfg, true) == ESP_OK) {
        uint16_t ap_count = 0;
        esp_wifi_scan_get_ap_num(&ap_count);
        if (ap_count > 0) {
            found_laptop = true;
        }
    }

    wifi_config_t sta_config;
    memset(&sta_config, 0, sizeof(sta_config));
    sta_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    if (found_laptop) {
        s_using_laptop_hotspot = true;
        strncpy((char *)sta_config.sta.ssid, LAPTOP_HOTSPOT_SSID, sizeof(sta_config.sta.ssid));
        strncpy((char *)sta_config.sta.password, LAPTOP_HOTSPOT_PASS, sizeof(sta_config.sta.password));
        apply_static_ip(LAPTOP_STATIC_IP, LAPTOP_GATEWAY_IP, LAPTOP_NETMASK);
        ESP_LOGI(TAG, "Hotspot de Laptop detectado (%s). Conectando directo...", LAPTOP_HOTSPOT_SSID);
    } else {
        s_using_laptop_hotspot = false;
        strncpy((char *)sta_config.sta.ssid, ROUTER_WIFI_SSID, sizeof(sta_config.sta.ssid));
        strncpy((char *)sta_config.sta.password, ROUTER_WIFI_PASS, sizeof(sta_config.sta.password));
        apply_static_ip(ROUTER_STATIC_IP, ROUTER_GATEWAY_IP, ROUTER_NETMASK);
        ESP_LOGI(TAG, "Conectando a red Wi-Fi (%s) con IP estatica %s...", ROUTER_WIFI_SSID, ROUTER_STATIC_IP);
    }

    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_config));
    esp_wifi_connect();
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGW(TAG, "Reconectando Wi-Fi...");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "==========================================================");
        ESP_LOGI(TAG, "  ESP32 MOTORES CONECTADO (%s)", s_using_laptop_hotspot ? "HOTSPOT LAPTOP DIRECTO" : "ROUTER LOCAL");
        ESP_LOGI(TAG, "  Control Remoto Web: http://" IPSTR "/", IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "==========================================================");
    }
}

void wifi_init_sta_pure(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    s_sta_netif = esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    /* CRÍTICO: Modo WIFI_MODE_STA puro (sin AP simultáneo) para no interferir con la ESP32-CAM */
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(60));

    select_and_connect_wifi();
}

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    init_motors();
    xTaskCreate(watchdog_task, "watchdog_task", 2048, NULL, 5, NULL);
    wifi_init_sta_pure();
    start_webserver();
}
