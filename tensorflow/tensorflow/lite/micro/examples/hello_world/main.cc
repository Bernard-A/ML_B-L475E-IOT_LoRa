/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stdio.h>

#include "lorawan/LoRaWANInterface.h"
#include "lorawan/system/lorawan_data_structures.h"
#include "events/EventQueue.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/examples/hello_world/constants.h"
#include "tensorflow/lite/micro/examples/hello_world/model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Application helpers
#include "lora_radio_helper.h"

using namespace events;

// Max payload size can be LORAMAC_PHY_MAXPAYLOAD.
// This example only communicates with much shorter messages (<30 bytes).
// If longer messages are used, these buffers must be changed accordingly.
uint8_t tx_buffer[30];
uint8_t rx_buffer[30];

/*
 * Setting up transmission threshold
 */
#define THRESHOLD			0.1

#define TX_INTERVAL             3000
#define MINIMUM_CONFIDENCE      0.7

/*
 * Sets up an application dependent transmission timer in ms. Used only when Duty Cycling is off for testing
 */
#define TX_TIMER                        10000

/**
 * Maximum number of events for the event queue.
 * 10 is the safe number for the stack events, however, if application
 * also uses the queue for whatever purposes, this number should be increased.
 */
#define MAX_NUMBER_OF_EVENTS            10

/**
 * Maximum number of retries for CONFIRMED messages before giving up
 */
#define CONFIRMED_MSG_RETRY_COUNTER     3

/**
* This event queue is the global event queue for both the
* application and stack. To conserve memory, the stack is designed to run
* in the same thread as the application and the application is responsible for
* providing an event queue to the stack that will be used for ISR deferment as
* well as application information event queuing.
*/
static EventQueue ev_queue(MAX_NUMBER_OF_EVENTS *EVENTS_EVENT_SIZE);

/**
 * Event handler.
 *
 * This will be passed to the LoRaWAN stack to queue events for the
 * application which in turn drive the application.
 */
static void lora_event_handler(lorawan_event_t event);

/**
 * Constructing Mbed LoRaWANInterface and passing it the radio object from lora_radio_helper.
 */
static LoRaWANInterface lorawan(radio);

/**
 * Application specific callbacks
 */
static lorawan_app_callbacks_t callbacks;

/**
 * Setting up TensorFlow Lite
 */
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Minimum arena size, at the time of writing. After allocating tensors
// you can retrieve this value by invoking interpreter.arena_used_bytes().
const int kModelArenaSize = 754;
// Extra headroom for model + alignment + future interpreter changes.
const int kExtraArenaSize = 554 + 16 + 100;
const int kTensorArenaSize = kModelArenaSize + kExtraArenaSize;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

/**
 * Entry point for application
 */
int main(void) {

    printf("Hello\n\n");
    // Setting up TF model
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Model provided is schema version %d not equal "
                           "to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
      return -1;
    }

    // This pulls in all the operation implementations we need.
    // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::AllOpsResolver resolver;
  
    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    printf("Arena_Used_Bytes : %d \n\n",interpreter->arena_used_bytes());

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return -1;
    }
    
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Keep track of how many inferences we have performed.
    inference_count = 0;

    // stores the status of a call to LoRaWAN protocol
    lorawan_status_t retcode;

    // Initialize LoRaWAN stack
    if (lorawan.initialize(&ev_queue) != LORAWAN_STATUS_OK) {
        printf("\r\n LoRa initialization failed! \r\n");
        return -1;
    }

    printf("\r\n Mbed LoRaWANStack initialized \r\n");

    // prepare application callbacks
    callbacks.events = mbed::callback(lora_event_handler);
    lorawan.add_app_callbacks(&callbacks);

    // Set number of retries in case of CONFIRMED messages
    if (lorawan.set_confirmed_msg_retries(CONFIRMED_MSG_RETRY_COUNTER)
            != LORAWAN_STATUS_OK) {
        printf("\r\n set_confirmed_msg_retries failed! \r\n\r\n");
        return -1;
    }

    printf("\r\n CONFIRMED message retries : %d \r\n",
           CONFIRMED_MSG_RETRY_COUNTER);

    // Enable adaptive data rate
    if (lorawan.enable_adaptive_datarate() != LORAWAN_STATUS_OK) {
        printf("\r\n enable_adaptive_datarate failed! \r\n");
        return -1;
    }

    printf("\r\n Adaptive data  rate (ADR) - Enabled \r\n");

    retcode = lorawan.connect();

    if (retcode == LORAWAN_STATUS_OK ||
            retcode == LORAWAN_STATUS_CONNECT_IN_PROGRESS) {
    } else {
        printf("\r\n Connection error, code = %d \r\n", retcode);
        return -1;
    }

    printf("\r\n Connection - In Progress ...\r\n");

    // make your event queue dispatching events forever
    ev_queue.dispatch_forever();

    return 0;      
}

/**
 * Sends a message to the Network Server
 */
static void send_message()
{
    // 0. Buffer allocation
    static uint8_t last_top_result = 99;
    static bool first_send_message = true;
    bool predict_nok;
    static float previously_transmitted;
    int index;

    // Loading data for tests if first time booting
    if (first_send_message) {
            previously_transmitted = conso_data[0];//load_data_Init();
            first_send_message = false;
            predict_nok = true;
            index = 0;
    }

    // 1. Import data into buffer
    float x_min = -363.16381836;
    float x_max = 373.3527832;
    float tx_min = 0.;
    float tx_max = 0.9;
    // Change this value to fit value for inferencing
    float x_val = (previously_transmitted - x_min) / (x_max - x_min) * (tx_max - tx_min) + tx_min;
            // X_min= -363.16381836
            // X_max= +373.3527832
            // max=0.9
            // min=0
            // Formula : X_scaled = (X-X_min)/(X_max-X_min) * (max-min) + min

    // Dual prediction
    // We position ourselves as the server and base prediction on previously transmitted value

    // Quantize the input from floating-point to integer
    int8_t x_quantized = x_val / input->params.scale + input->params.zero_point;
    // Place the quantized input in the model's input tensor
    input->data.int8[0] = x_quantized;


    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                           static_cast<double>(x_val));
      return;
    }
    
    // Obtain the quantized output from model's output tensor
    int8_t y_quantized = output->data.int8[0];
    // Dequantize the output from integer to floating-point
    float y_val = (y_quantized - output->params.zero_point) * output->params.scale;

    // Determining difference unscaled
    y_val = (y_val - tx_min) / (tx_max - tx_min) * (x_max - x_min) + x_min + previously_transmitted;

    index++;

    if (index == 3) { index = 0;}

    float difference_prediction = (y_val-conso_data[index])/(conso_data[index]);

    if (difference_prediction < 0) {
            difference_prediction = - difference_prediction;
    }

    if (difference_prediction < THRESHOLD) {
        predict_nok=false;
    } else {
        predict_nok=true;
        y_val=conso_data[index];
    }

    // 3. Log the prediction

    printf("%f",y_val);

    // 4. Send data based on prediction

    previously_transmitted = y_val;

    uint16_t packet_len;
    int16_t retcode;

    if (predict_nok) {
        packet_len = sprintf((char *) tx_buffer, "Transmitted Value is %f",
            y_val);
        retcode = lorawan.send(MBED_CONF_LORA_APP_PORT, tx_buffer, packet_len,
                           MSG_UNCONFIRMED_FLAG);

        if (retcode < 0) {
            retcode == LORAWAN_STATUS_WOULD_BLOCK ? printf("send - Duty cycle violation\r\n")
                : printf("send - Error code %d \r\n", retcode);

            if (retcode == LORAWAN_STATUS_WOULD_BLOCK) {
                // retry in 3 secs
                ev_queue.call_in(3000, send_message);
            }
            return;
        }
        printf("%d bytes scheduled for transmission \r\n", retcode);
        memset(tx_buffer, 0, sizeof(tx_buffer));
    }
}

/**
 * Receive a message from the Network Server
 */
static void receive_message()
{
    uint8_t port;
    int flags;
    int16_t retcode = lorawan.receive(rx_buffer, sizeof(rx_buffer), port, flags);

    if (retcode < 0) {
        printf("\r\n receive() - Error code %d \r\n", retcode);
        return;
    }

    printf(" RX Data on port %u (%d bytes): ", port, retcode);
    for (uint8_t i = 0; i < retcode; i++) {
        printf("%02x ", rx_buffer[i]);
    }
    printf("\r\n");

    memset(rx_buffer, 0, sizeof(rx_buffer));
}

/**
 * Event handler
 */
static void lora_event_handler(lorawan_event_t event)
{
    switch (event) {
        case CONNECTED:
            printf("\r\n Connection - Successful \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            } else {
                ev_queue.call_every(TX_TIMER, send_message);
            }

            break;
        case DISCONNECTED:
            ev_queue.break_dispatch();
            printf("\r\n Disconnected Successfully \r\n");
            break;
        case TX_DONE:
            printf("\r\n Message Sent to Network Server \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        case TX_TIMEOUT:
        case TX_ERROR:
        case TX_CRYPTO_ERROR:
        case TX_SCHEDULING_ERROR:
            printf("\r\n Transmission Error - EventCode = %d \r\n", event);
            // try again
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        case RX_DONE:
            printf("\r\n Received message from Network Server \r\n");
            receive_message();
            break;
        case RX_TIMEOUT:
        case RX_ERROR:
            printf("\r\n Error in reception - Code = %d \r\n", event);
            break;
        case JOIN_FAILURE:
            printf("\r\n OTAA Failed - Check Keys \r\n");
            break;
        case UPLINK_REQUIRED:
            printf("\r\n Uplink required by NS \r\n");
            if (MBED_CONF_LORA_DUTY_CYCLE_ON) {
                send_message();
            }
            break;
        default:
            MBED_ASSERT("Unknown Event");
    }
}

//EOF
