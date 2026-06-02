// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2026 @malonestar
// Wake-word provider, adapted from ESPHome's micro_wake_word component
// (github.com/esphome/esphome, (c) 2019 ESPHome) — GPLv3. See LICENSE.

#include "micro_wake_word.h"
#include "audio_service.h"
#include <esp_log.h>
#include <esp_heap_caps.h>
#include "settings.h"
#include <esp_system.h>

#define DETECTION_RUNNING_EVENT 1
#define MWW_NVS_NS    "mww"
#define MWW_NVS_KEY   "model"
#define TAG "MicroWakeWord"

// All three models are embedded (see main/CMakeLists.txt EMBED_FILES). One is
// loaded at a time; selection persists in NVS (namespace "mww", key "model").
extern const uint8_t hey_stackchan_v1_start[] asm("_binary_hey_stackchan_v1_tflite_start");
extern const uint8_t hey_frank_v5_start[]     asm("_binary_hey_frank_v5_tflite_start");
extern const uint8_t hey_m5_v3_start[]        asm("_binary_hey_m5_v3_tflite_start");

const MicroWakeWord::ModelEntry MicroWakeWord::kModels[MicroWakeWord::kModelCount] = {
    // cutoffs tuned on-device; the .json probability_cutoff values were conservative.
    { "stackchan", "hey_stackchan_v1", hey_stackchan_v1_start, 217, "Hey,Stack Chan" },
    { "frank",     "hey_frank_v5",     hey_frank_v5_start,     204, "Hey,Frank"     },
    { "m5",        "hey_m5_v3",        hey_m5_v3_start,        217, "Hey,M5"        },
};

int MicroWakeWord::ResolveModelId(const std::string& id) const {
    for (size_t i = 0; i < kModelCount; ++i) {
        if (id == kModels[i].id) return static_cast<int>(i);
    }
    return -1;
}

std::string MicroWakeWord::LoadPersistedModelId() {
    Settings s(MWW_NVS_NS, false);
    return s.GetString(MWW_NVS_KEY, kDefaultModelId);
}

MicroWakeWord::MicroWakeWord() {
    event_group_ = xEventGroupCreate();
    active_index_ = ResolveModelId(LoadPersistedModelId());
    if (active_index_ < 0) active_index_ = ResolveModelId(kDefaultModelId);
    if (active_index_ < 0) active_index_ = 0;
    model_start_ = kModels[active_index_].start;
    cutoff_ = kModels[active_index_].cutoff;
    phrase_ = kModels[active_index_].phrase;
    recent_probabilities_.resize(SLIDING_WINDOW_SIZE, 0);
    ignore_windows_ = -MIN_SLICES_BEFORE_DETECTION;
}

MicroWakeWord::~MicroWakeWord() {
    interpreter_.reset();
    if (tensor_arena_) heap_caps_free(tensor_arena_);
    if (var_arena_) heap_caps_free(var_arena_);
    if (frontend_initialized_) FrontendFreeStateContents(&frontend_state_);
    if (wake_word_encode_task_stack_) heap_caps_free(wake_word_encode_task_stack_);
    if (wake_word_encode_task_buffer_) heap_caps_free(wake_word_encode_task_buffer_);
    vEventGroupDelete(event_group_);
}

bool MicroWakeWord::RegisterOps() {
    if (ops_registered_) return true;
    if (op_resolver_.AddCallOnce() != kTfLiteOk) return false;
    if (op_resolver_.AddVarHandle() != kTfLiteOk) return false;
    if (op_resolver_.AddReshape() != kTfLiteOk) return false;
    if (op_resolver_.AddReadVariable() != kTfLiteOk) return false;
    if (op_resolver_.AddStridedSlice() != kTfLiteOk) return false;
    if (op_resolver_.AddConcatenation() != kTfLiteOk) return false;
    if (op_resolver_.AddAssignVariable() != kTfLiteOk) return false;
    if (op_resolver_.AddConv2D() != kTfLiteOk) return false;
    if (op_resolver_.AddMul() != kTfLiteOk) return false;
    if (op_resolver_.AddAdd() != kTfLiteOk) return false;
    if (op_resolver_.AddMean() != kTfLiteOk) return false;
    if (op_resolver_.AddFullyConnected() != kTfLiteOk) return false;
    if (op_resolver_.AddLogistic() != kTfLiteOk) return false;
    if (op_resolver_.AddQuantize() != kTfLiteOk) return false;
    if (op_resolver_.AddDepthwiseConv2D() != kTfLiteOk) return false;
    if (op_resolver_.AddAveragePool2D() != kTfLiteOk) return false;
    if (op_resolver_.AddMaxPool2D() != kTfLiteOk) return false;
    if (op_resolver_.AddPad() != kTfLiteOk) return false;
    if (op_resolver_.AddPack() != kTfLiteOk) return false;
    if (op_resolver_.AddSplitV() != kTfLiteOk) return false;
    ops_registered_ = true;
    return true;
}

bool MicroWakeWord::LoadModel() {
    auto cleanup_and_fail = [this]() -> bool {
        interpreter_.reset();
        if (tensor_arena_) { heap_caps_free(tensor_arena_); tensor_arena_ = nullptr; }
        if (var_arena_)    { heap_caps_free(var_arena_);    var_arena_ = nullptr; }
        ma_ = nullptr; mrv_ = nullptr;
        return false;
    };

    if (!RegisterOps()) {
        ESP_LOGE(TAG, "Failed to register TFLite ops");
        return cleanup_and_fail();
    }

    var_arena_ = (uint8_t*)heap_caps_malloc(VARIABLE_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!var_arena_) {
        ESP_LOGE(TAG, "Failed to allocate variable arena");
        return cleanup_and_fail();
    }
    ma_ = tflite::MicroAllocator::Create(var_arena_, VARIABLE_ARENA_SIZE);
    mrv_ = tflite::MicroResourceVariables::Create(ma_, 20);

    tensor_arena_ = (uint8_t*)heap_caps_malloc(TENSOR_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena (%d bytes)", TENSOR_ARENA_SIZE);
        return cleanup_and_fail();
    }

    interpreter_ = std::make_unique<tflite::MicroInterpreter>(
        tflite::GetModel(model_start_), op_resolver_,
        tensor_arena_, TENSOR_ARENA_SIZE, mrv_);

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors (arena=%d bytes)", TENSOR_ARENA_SIZE);
        return cleanup_and_fail();
    }

    ESP_LOGI(TAG, "Tensor arena: %u of %d bytes used",
             (unsigned)interpreter_->arena_used_bytes(), TENSOR_ARENA_SIZE);

    TfLiteTensor* input = interpreter_->input(0);
    if (input->dims->size != 3 || input->dims->data[0] != 1 ||
        input->dims->data[2] != PREPROCESSOR_FEATURE_SIZE) {
        ESP_LOGE(TAG, "Model input dimensions mismatch: %dx%dx%d",
                 input->dims->data[0], input->dims->data[1], input->dims->data[2]);
        return cleanup_and_fail();
    }

    ESP_LOGI(TAG, "Model loaded, stride=%d, arena=%d bytes",
             input->dims->data[1], TENSOR_ARENA_SIZE);
    return true;
}

bool MicroWakeWord::Initialize(AudioCodec* codec, srmodel_list_t* models_list) {
    codec_ = codec;
    input_channels_ = codec_->input_channels();
    ESP_LOGI(TAG, "Codec input channels: %d", input_channels_);

    frontend_config_.window.size_ms = FEATURE_DURATION_MS;
    frontend_config_.window.step_size_ms = features_step_size_;
    frontend_config_.filterbank.num_channels = PREPROCESSOR_FEATURE_SIZE;
    frontend_config_.filterbank.lower_band_limit = 125.0;
    frontend_config_.filterbank.upper_band_limit = 7500.0;
    frontend_config_.noise_reduction.smoothing_bits = 10;
    frontend_config_.noise_reduction.even_smoothing = 0.10;
    frontend_config_.noise_reduction.odd_smoothing = 0.15;
    frontend_config_.noise_reduction.min_signal_remaining = 0.05;
    frontend_config_.pcan_gain_control.enable_pcan = true;
    frontend_config_.pcan_gain_control.strength = 0.95;
    frontend_config_.pcan_gain_control.offset = 80.0;
    frontend_config_.pcan_gain_control.gain_bits = 21;
    frontend_config_.log_scale.enable_log = true;
    frontend_config_.log_scale.scale_shift = 6;

    if (!FrontendPopulateState(&frontend_config_, &frontend_state_, 16000)) {
        ESP_LOGE(TAG, "Failed to initialize audio frontend");
        return false;
    }
    frontend_initialized_ = true;

    if (!LoadModel()) {
        ESP_LOGE(TAG, "Failed to load TFLite model");
        return false;
    }

    xTaskCreate([](void* arg) {
        auto self = (MicroWakeWord*)arg;
        self->AudioDetectionTask();
        vTaskDelete(NULL);
    }, "mww_detect", 4096, this, 3, nullptr);

    ESP_LOGI(TAG, "Initialized with model %s, cutoff=%.2f",
             kModels[active_index_].display, cutoff_ / 255.0f);
    return true;
}

void MicroWakeWord::OnWakeWordDetected(std::function<void(const std::string& wake_word)> callback) {
    wake_word_detected_callback_ = callback;
}

void MicroWakeWord::Start() {
    if (interpreter_) interpreter_->Reset();
    current_stride_step_ = 0;
    last_n_index_ = 0;
    for (auto& p : recent_probabilities_) p = 0;
    ignore_windows_ = -MIN_SLICES_BEFORE_DETECTION;
    if (frontend_initialized_) FrontendReset(&frontend_state_);
    xEventGroupSetBits(event_group_, DETECTION_RUNNING_EVENT);
}

void MicroWakeWord::Stop() {
    xEventGroupClearBits(event_group_, DETECTION_RUNNING_EVENT);
    std::lock_guard<std::mutex> lock(input_buffer_mutex_);
    input_buffer_.clear();
}

void MicroWakeWord::Feed(const std::vector<int16_t>& data) {
    std::lock_guard<std::mutex> lock(input_buffer_mutex_);
    if (!(xEventGroupGetBits(event_group_) & DETECTION_RUNNING_EVENT)) {
        return;
    }
    if (input_channels_ > 1) {
        size_t mono_samples = data.size() / input_channels_;
        input_buffer_.reserve(input_buffer_.size() + mono_samples);
        for (size_t i = 0; i < mono_samples; ++i) {
            input_buffer_.push_back(data[i * input_channels_]);
        }
    } else {
        input_buffer_.insert(input_buffer_.end(), data.begin(), data.end());
    }
}

size_t MicroWakeWord::GetFeedSize() {
    return 16000 * features_step_size_ / 1000;
}

bool MicroWakeWord::GenerateFeatures(const int16_t* audio, size_t samples_available,
                                      int8_t* features_out, size_t* processed_samples) {
    *processed_samples = 0;
    struct FrontendOutput frontend_output =
        FrontendProcessSamples(&frontend_state_, audio, samples_available, processed_samples);

    if (frontend_output.size == 0) return false;

    for (size_t i = 0; i < frontend_output.size; ++i) {
        constexpr int32_t value_scale = 256;
        constexpr int32_t value_div = 666;
        int32_t value = ((frontend_output.values[i] * value_scale) + (value_div / 2)) / value_div;
        value += INT8_MIN;
        if (value < INT8_MIN) value = INT8_MIN;
        if (value > INT8_MAX) value = INT8_MAX;
        features_out[i] = static_cast<int8_t>(value);
    }
    return true;
}

void MicroWakeWord::AudioDetectionTask() {
    ESP_LOGI(TAG, "Detection task started");
    int8_t features[PREPROCESSOR_FEATURE_SIZE];
    uint32_t feed_count = 0;
    uint32_t feature_count = 0;
    uint32_t inference_count = 0;
    uint8_t max_prob_seen = 0;

    while (true) {
        xEventGroupWaitBits(event_group_, DETECTION_RUNNING_EVENT, pdFALSE, pdTRUE, portMAX_DELAY);

        std::vector<int16_t> local_buf;
        {
            std::lock_guard<std::mutex> lock(input_buffer_mutex_);
            if (input_buffer_.empty()) {
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            local_buf.swap(input_buffer_);
        }

        ++feed_count;
        if (feed_count % 100 == 0) {
            ESP_LOGI(TAG, "[DBG] feeds=%lu features=%lu inferences=%lu max_prob=%u",
                     (unsigned long)feed_count, (unsigned long)feature_count,
                     (unsigned long)inference_count, max_prob_seen);
            max_prob_seen = 0;
        }

        StoreWakeWordData(local_buf.data(), local_buf.size());

        size_t offset = 0;
        while (offset < local_buf.size()) {
            size_t processed = 0;
            bool got_feature = GenerateFeatures(
                local_buf.data() + offset,
                local_buf.size() - offset,
                features, &processed);

            if (processed == 0) break;
            offset += processed;
            if (!got_feature) continue;
            ++feature_count;

            TfLiteTensor* input = interpreter_->input(0);
            uint8_t stride = input->dims->data[1];
            current_stride_step_ = current_stride_step_ % stride;

            std::memmove(
                tflite::GetTensorData<int8_t>(input) + PREPROCESSOR_FEATURE_SIZE * current_stride_step_,
                features, PREPROCESSOR_FEATURE_SIZE);
            ++current_stride_step_;

            if (current_stride_step_ >= stride) {
                if (interpreter_->Invoke() != kTfLiteOk) {
                    ESP_LOGE(TAG, "Inference failed");
                    continue;
                }

                ++inference_count;
                TfLiteTensor* output = interpreter_->output(0);
                uint8_t probability = output->data.uint8[0];
                if (probability > max_prob_seen) max_prob_seen = probability;

                ++last_n_index_;
                if (last_n_index_ == SLIDING_WINDOW_SIZE) last_n_index_ = 0;
                recent_probabilities_[last_n_index_] = probability;

                // Warm-up gate: advance once per inference window, unconditionally.
                // (ESPHome StreamingModel does this every window. The original port
                // gated it on probability<cutoff, so a strong wake utterance during
                // warm-up stalled the counter and detection never armed — a perfect
                // 255-avg utterance would sail through still gated off.)
                ignore_windows_ = std::min<int16_t>(ignore_windows_ + 1, 0);
            }

            if (ignore_windows_ < 0) continue;

            uint32_t sum = 0;
            for (auto& p : recent_probabilities_) sum += p;

            if (sum > (uint32_t)cutoff_ * SLIDING_WINDOW_SIZE) {
                ESP_LOGI(TAG, "Wake word detected! avg=%.2f",
                         (sum / (float)SLIDING_WINDOW_SIZE) / 255.0f);
                Stop();
                last_detected_wake_word_ = phrase_;

                for (auto& p : recent_probabilities_) p = 0;
                ignore_windows_ = -MIN_SLICES_BEFORE_DETECTION;

                if (wake_word_detected_callback_) {
                    wake_word_detected_callback_(last_detected_wake_word_);
                }
            }
        }
    }
}

void MicroWakeWord::StoreWakeWordData(const int16_t* data, size_t samples) {
    wake_word_pcm_.emplace_back(std::vector<int16_t>(data, data + samples));
    while (wake_word_pcm_.size() > 2000 / 30) {
        wake_word_pcm_.pop_front();
    }
}

void MicroWakeWord::EncodeWakeWordData() {
    const size_t stack_size = 4096 * 6;
    wake_word_opus_.clear();
    if (wake_word_encode_task_stack_ == nullptr) {
        wake_word_encode_task_stack_ = (StackType_t*)heap_caps_malloc(stack_size, MALLOC_CAP_SPIRAM);
        assert(wake_word_encode_task_stack_ != nullptr);
    }
    if (wake_word_encode_task_buffer_ == nullptr) {
        wake_word_encode_task_buffer_ = (StaticTask_t*)heap_caps_malloc(sizeof(StaticTask_t), MALLOC_CAP_INTERNAL);
        assert(wake_word_encode_task_buffer_ != nullptr);
    }

    wake_word_encode_task_ = xTaskCreateStatic([](void* arg) {
        auto self = (MicroWakeWord*)arg;
        {
            auto start_time = esp_timer_get_time();
            esp_opus_enc_config_t opus_enc_cfg = AS_OPUS_ENC_CONFIG();
            void* encoder_handle = nullptr;
            auto ret = esp_opus_enc_open(&opus_enc_cfg, sizeof(esp_opus_enc_config_t), &encoder_handle);
            if (encoder_handle == nullptr) {
                ESP_LOGE(TAG, "Failed to create audio encoder: %d", ret);
                std::lock_guard<std::mutex> lock(self->wake_word_mutex_);
                self->wake_word_opus_.push_back(std::vector<uint8_t>());
                self->wake_word_cv_.notify_all();
                return;
            }

            int frame_size = 0;
            int outbuf_size = 0;
            esp_opus_enc_get_frame_size(encoder_handle, &frame_size, &outbuf_size);
            frame_size = frame_size / sizeof(int16_t);

            int packets = 0;
            std::vector<int16_t> in_buffer;
            esp_audio_enc_in_frame_t in = {};
            esp_audio_enc_out_frame_t out = {};

            for (auto& pcm : self->wake_word_pcm_) {
                if (in_buffer.empty()) {
                    in_buffer = std::move(pcm);
                } else {
                    in_buffer.reserve(in_buffer.size() + pcm.size());
                    in_buffer.insert(in_buffer.end(), pcm.begin(), pcm.end());
                }

                while (in_buffer.size() >= (size_t)frame_size) {
                    std::vector<uint8_t> opus_buf(outbuf_size);
                    in.buffer = (uint8_t*)(in_buffer.data());
                    in.len = (uint32_t)(frame_size * sizeof(int16_t));
                    out.buffer = opus_buf.data();
                    out.len = outbuf_size;
                    out.encoded_bytes = 0;

                    ret = esp_opus_enc_process(encoder_handle, &in, &out);
                    if (ret == ESP_AUDIO_ERR_OK) {
                        std::lock_guard<std::mutex> lock(self->wake_word_mutex_);
                        self->wake_word_opus_.emplace_back(opus_buf.data(), opus_buf.data() + out.encoded_bytes);
                        self->wake_word_cv_.notify_all();
                        packets++;
                    }
                    in_buffer.erase(in_buffer.begin(), in_buffer.begin() + frame_size);
                }
            }
            self->wake_word_pcm_.clear();
            esp_opus_enc_close(encoder_handle);
            auto end_time = esp_timer_get_time();
            ESP_LOGI(TAG, "Encoded wake word: %d opus packets in %ld ms",
                     packets, (long)((end_time - start_time) / 1000));

            std::lock_guard<std::mutex> lock(self->wake_word_mutex_);
            self->wake_word_opus_.push_back(std::vector<uint8_t>());
            self->wake_word_cv_.notify_all();
        }
        vTaskDelete(NULL);
    }, "encode_wake_word", stack_size, this, 2, wake_word_encode_task_stack_, wake_word_encode_task_buffer_);
}

bool MicroWakeWord::GetWakeWordOpus(std::vector<uint8_t>& opus) {
    std::unique_lock<std::mutex> lock(wake_word_mutex_);
    wake_word_cv_.wait(lock, [this]() {
        return !wake_word_opus_.empty();
    });
    opus.swap(wake_word_opus_.front());
    wake_word_opus_.pop_front();
    return !opus.empty();
}
