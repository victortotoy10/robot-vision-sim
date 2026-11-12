idf_component_register(SRCS "main.c"
                       INCLUDE_DIRS "."
                       REQUIRES driver nvs_flash esp_wifi esp_http_server esp_event esp_netif esp_timer freertos esp_system log lwip)
