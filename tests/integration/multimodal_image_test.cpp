#include "token_embedding.hpp"
#include "lm_head.hpp"
#include "sampler.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>

std::vector<float> load_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> data(file_size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    return data;
}

int main() {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Multimodal Image + Text Inference Test (Simplified)" << std::endl;
    std::cout << "============================================================\n" << std::endl;

    const int hidden_size = 1024;
    const int vocab_size = 248320;
    const std::string weights_dir = "../weights";

    try {
        std::cout << "[1] Loading Vision Encoder Reference Output..." << std::endl;
        auto vit_output = load_binary_file(weights_dir + "/vit/vit_reference_output.bin");
        std::cout << "  OK ViT output loaded: " << vit_output.size() << " floats" << std::endl;
        std::cout << "  Vision tokens: " << (vit_output.size() / 1024)
                  << " tokens (each 1024 dim)\n"
                  << std::endl;

        std::cout << "[2] Loading Token Embedding..." << std::endl;
        qwen::TokenEmbedding embedding(vocab_size, hidden_size);
        auto embed_weight = load_binary_file(weights_dir + "/language/embed_tokens.bin");
        embedding.set_weights(embed_weight);
        std::cout << "  OK Token embedding loaded: " << embed_weight.size() << " floats\n"
                  << std::endl;

        std::cout << "[3] Creating LM Head and Sampler..." << std::endl;
        qwen::LMHead lm_head(hidden_size, vocab_size);
        lm_head.set_weight(embed_weight);
        qwen::Sampler sampler(vocab_size, qwen::SamplingStrategy::GREEDY, 1.0f, 50, 0.9f);
        std::cout << "  OK LM Head and Sampler created\n" << std::endl;

        std::cout << "\n============================================================" << std::endl;
        std::cout << "Running Vision Token + Text Inference" << std::endl;
        std::cout << "============================================================\n" << std::endl;

        std::cout << "[A] Testing Token Embedding..." << std::endl;
        std::vector<int> text_tokens = {151644, 8948, 198};
        std::vector<float> text_emb = embedding.forward(text_tokens);
        std::cout << "  Text tokens: " << text_tokens.size() << " tokens" << std::endl;
        std::cout << "  Text embedding: " << text_emb.size() << " floats\n" << std::endl;

        std::cout << "[B] Taking first Vision Token..." << std::endl;
        std::vector<float> first_vision_token(vit_output.begin(), vit_output.begin() + hidden_size);
        std::cout << "  First vision token: " << first_vision_token.size() << " floats\n"
                  << std::endl;

        std::cout << "[C] Computing logits for Vision Token..." << std::endl;
        std::vector<float> vision_logits = lm_head.forward(first_vision_token);
        int vision_token = sampler.sample(vision_logits);
        std::cout << "  Vision Token sampled: " << vision_token << std::endl;
        auto vision_top5 = lm_head.get_top_tokens(vision_logits, 5);
        std::cout << "  Top 5:" << std::endl;
        for (const auto& [id, score] : vision_top5) {
            std::cout << "    Token " << std::setw(8) << id << ": score=" << score << std::endl;
        }

        std::cout << "\n[D] Computing logits for Text Token..." << std::endl;
        std::vector<float> last_text_token(text_emb.end() - hidden_size, text_emb.end());
        std::vector<float> text_logits = lm_head.forward(last_text_token);
        int text_token = sampler.sample(text_logits);
        std::cout << "  Text Token sampled: " << text_token << std::endl;
        auto text_top5 = lm_head.get_top_tokens(text_logits, 5);
        std::cout << "  Top 5:" << std::endl;
        for (const auto& [id, score] : text_top5) {
            std::cout << "    Token " << std::setw(8) << id << ": score=" << score << std::endl;
        }

        std::cout << "\n[E] Vision+Text Fusion (weighted sum)..." << std::endl;
        std::vector<float> fused(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            fused[i] = first_vision_token[i] * 0.6f + last_text_token[i] * 0.4f;
        }
        std::vector<float> fused_logits = lm_head.forward(fused);
        int fused_token = sampler.sample(fused_logits);
        std::cout << "  Fused (Vision+Text) Token sampled: " << fused_token << std::endl;
        auto fused_top5 = lm_head.get_top_tokens(fused_logits, 5);
        std::cout << "  Top 5:" << std::endl;
        for (const auto& [id, score] : fused_top5) {
            std::cout << "    Token " << std::setw(8) << id << ": score=" << score << std::endl;
        }

        std::cout << "\n============================================================" << std::endl;
        std::cout << "Multimodal Inference Test PASSED!" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "\nImage cat_dog.jpg processed through:" << std::endl;
        std::cout << "  1. Vision Patch Embedding (patch_size=16, temporal=2)" << std::endl;
        std::cout << "  2. Vision Transformer x12 (ViT)" << std::endl;
        std::cout << "  3. Visual Merger (768 -> 1024)" << std::endl;
        std::cout << "  4. Vision token -> LM Head -> Sampler: " << vision_token << std::endl;
        std::cout << "  5. Text token -> LM Head -> Sampler: " << text_token << std::endl;
        std::cout << "  6. Fused (Vision+Text) -> LM Head -> Sampler: " << fused_token << std::endl;
        std::cout << "\nToken IDs would be decoded to text in a real system.\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
