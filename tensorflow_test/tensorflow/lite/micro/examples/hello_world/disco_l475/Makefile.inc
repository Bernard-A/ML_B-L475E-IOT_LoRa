# Settings for the Discovery STM32L475 board.
ifneq ($(filter disco_l475,$(ALL_TAGS)),)
  hello_world_MBED_PROJECT_FILES += \
    BSP_B-L475E-IOT01.lib
endif
