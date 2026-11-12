idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp32-camera esp_http_server esp_wifi esp_event esp_netif esp_timer driver nvs_flash freertos esp_system log heap lwip)

