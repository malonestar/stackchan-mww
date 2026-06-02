// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 @malonestar
// Wake-word provider, adapted from ESPHome's micro_wake_word component
// (github.com/esphome/esphome, (c) 2019 ESPHome) — GPLv3. See LICENSE.

#ifndef MICRO_WAKE_WORD_H
#define MICRO_WAKE_WORD_H

#include <freertos/FreeRTOS.h>
#include <freertos/event_groups.h>

#include <deque>
#include <string>
#include <vector>
#include <functional>
#include <mutex>
#include <condition_variable>

#include <frontend.h>
#include <frontend_util.h>

#include <tensorflow/lite/core/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>

#include "audio_codec.h"
#include "wake_word.h"

class MicroWakeWord : public WakeWord {
public:
    MicroWakeWord();
    ~MicroWakeWord();

    bool Initialize(AudioCodec* codec, srmodel_list_t* models_list) override;
    void Feed(const std::vector<int16_t>& data) override;
    void OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback) override;
    void Start() override;
    void Stop() override;
    size_t GetFeedSize() override;
    void EncodeWakeWordData() override;
    bool GetWakeWordOpus(std::vector<uint8_t>& opus) override;
    const std::string& GetLastDetectedWakeWord() const override { return last_detected_wake_word_; }

private:
    static constexpr uint8_t PREPROCESSOR_FEATURE_SIZE = 40;
    static constexpr uint8_t FEATURE_DURATION_MS = 30;
    static constexpr size_t SLIDING_WINDOW_SIZE = 5;
    static constexpr size_t TENSOR_ARENA_SIZE = 65536;
    static constexpr size_t VARIABLE_ARENA_SIZE = 1024;
    static constexpr uint8_t MIN_SLICES_BEFORE_DETECTION = 100;

    // One embedded wake-word model and its runtime parameters.
    struct ModelEntry {
        const char*    id;       // NVS-safe id: "stackchan" | "frank" | "m5"
        const char*    display;  // human label, e.g. "hey_stackchan_v1"
        const uint8_t* start;    // _binary_<name>_tflite_start
        uint8_t        cutoff;   // probability threshold (0-255)
        const char*    phrase;   // wake phrase reported on detection
    };
    static constexpr size_t kModelCount = 3;
    static const ModelEntry kModels[kModelCount];
    static constexpr const char* kDefaultModelId = "m5";

    // Resolve an id to a kModels index; returns -1 if unknown.
    int ResolveModelId(const std::string& id) const;

    EventGroupHandle_t event_group_;
    std::function<void(const std::string& wake_word)> wake_word_detected_callback_;
    AudioCodec* codec_ = nullptr;
    std::string last_detected_wake_word_;
    int input_channels_ = 1;

    std::vector<int16_t> input_buffer_;
    std::mutex input_buffer_mutex_;

    struct FrontendConfig frontend_config_;
    struct FrontendState frontend_state_;
    bool frontend_initialized_ = false;
    uint8_t features_step_size_ = 10;

    const uint8_t* model_start_;
    int active_index_ = 0;     // index into kModels of the loaded model
    uint8_t cutoff_ = 207;     // active model's probability cutoff
    std::string phrase_;       // active model's wake phrase
    bool ops_registered_ = false; // RegisterOps runs once; guards re-entry
    uint8_t* tensor_arena_ = nullptr;
    uint8_t* var_arena_ = nullptr;
    std::unique_ptr<tflite::MicroInterpreter> interpreter_;
    tflite::MicroMutableOpResolver<20> op_resolver_;
    tflite::MicroResourceVariables* mrv_ = nullptr;
    tflite::MicroAllocator* ma_ = nullptr;

    std::vector<uint8_t> recent_probabilities_;
    size_t last_n_index_ = 0;
    uint8_t current_stride_step_ = 0;
    int16_t ignore_windows_ = 0;

    TaskHandle_t wake_word_encode_task_ = nullptr;
    StaticTask_t* wake_word_encode_task_buffer_ = nullptr;
    StackType_t* wake_word_encode_task_stack_ = nullptr;
    std::deque<std::vector<int16_t>> wake_word_pcm_;
    std::deque<std::vector<uint8_t>> wake_word_opus_;
    std::mutex wake_word_mutex_;
    std::condition_variable wake_word_cv_;

    bool LoadModel();
    std::string LoadPersistedModelId();
    bool RegisterOps();
    bool GenerateFeatures(const int16_t* audio, size_t samples_available,
                          int8_t* features_out, size_t* processed_samples);
    void StoreWakeWordData(const int16_t* data, size_t samples);
    void AudioDetectionTask();
};

#endif
