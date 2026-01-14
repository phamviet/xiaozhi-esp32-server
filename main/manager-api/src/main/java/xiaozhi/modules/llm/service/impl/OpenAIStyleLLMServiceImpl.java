package xiaozhi.modules.llm.service.impl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import cn.hutool.json.JSONArray;
import cn.hutool.json.JSONObject;
import cn.hutool.json.JSONUtil;
import lombok.extern.slf4j.Slf4j;
import xiaozhi.modules.llm.service.LLMService;
import xiaozhi.modules.model.entity.ModelConfigEntity;
import xiaozhi.modules.model.service.ModelConfigService;

/**
 * OpenAI风格API的LLM服务实现
 * 支持阿里云、DeepSeek、ChatGLM等兼容OpenAI API的模型
 */
@Slf4j
@Service
public class OpenAIStyleLLMServiceImpl implements LLMService {

    @Autowired
    private ModelConfigService modelConfigService;

    private final RestTemplate restTemplate = new RestTemplate();

    private static final String DEFAULT_SUMMARY_PROMPT = """
        [SYSTEM Response in user language which is determined from `New Conversation Content`]

        You are an experienced memory summarizer, skilled at summarizing conversation content. Please follow these rules:

        1. Summarize key user information to provide more personalized service in future conversations.

        2. Avoid repetitive summaries and do not forget previous memories unless the original memory exceeds 1800 words. Do not forget or compress the user's historical memory.

        3. User-controlled device volume, music playback, weather, exiting the conversation, or indicating unwillingness to continue the conversation—information unrelated to the user—do not need to be included in the summary.

        4. Today's date, time, and weather information—data unrelated to the user's events—do not need to be included in the summary if stored as memory, as this could affect subsequent conversations.

        5. Do not include device control... Include both the results and failures of the control measures in the summary and avoid including irrelevant user comments.

        6. Don't summarize for the sake of summarizing. If a user's chat is meaningless, reverting to the original history is acceptable.

        7. Only return a summary abstract, strictly limited to 1800 characters.

        8. Do not include code or XML. No explanations, comments, or descriptions are needed. When saving the memory, only extract information from the conversation; do not mix in example content.

        9. If a history memory is provided, intelligently merge the new conversation content with the history memory, retaining valuable historical information while adding new important information. History Memory:

        {history_memory}

        New Conversation Content: {conversation}
    """;

    @Override
    public String generateSummary(String conversation) {
        return generateSummary(conversation, null, null);
    }

    @Override
    public String generateSummaryWithModel(String conversation, String modelId) {
        return generateSummary(conversation, null, modelId);
    }

    @Override
    public String generateSummary(String conversation, String promptTemplate, String modelId) {
        if (!isAvailable()) {
            log.warn("LLM service unavailable, unable to generate summary");
            return "LLM service unavailable, unable to generate summary";
        }

        try {
            // 从智控台获取LLM模型配置
            ModelConfigEntity llmConfig;
            if (modelId != null && !modelId.trim().isEmpty()) {
                // 通过具体模型ID获取配置
                llmConfig = modelConfigService.getModelByIdFromCache(modelId);
            } else {
                // 保持向后兼容，使用默认配置
                llmConfig = getDefaultLLMConfig();
            }

            if (llmConfig == null || llmConfig.getConfigJson() == null) {
                log.error("No available LLM model configuration found，modelId: {}", modelId);
                return "No available LLM model configuration found";
            }

            JSONObject configJson = llmConfig.getConfigJson();
            String baseUrl = configJson.getStr("base_url");
            String model = configJson.getStr("model_name");
            String apiKey = configJson.getStr("api_key");
            Double temperature = configJson.getDouble("temperature");
            Integer maxTokens = configJson.getInt("max_tokens");

            if (StringUtils.isBlank(baseUrl) || StringUtils.isBlank(apiKey)) {
                log.error("LLM配置不完整，baseUrl或apiKey为空");
                return "The LLM configuration is incomplete, and a summary cannot be generated.";
            }

            // 构建提示词
            String prompt = (promptTemplate != null ? promptTemplate : DEFAULT_SUMMARY_PROMPT).replace("{conversation}",
                    conversation);

            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", model != null ? model : "gpt-3.5-turbo");

            Map<String, Object>[] messages = new Map[1];
            Map<String, Object> message = new HashMap<>();
            message.put("role", "user");
            message.put("content", prompt);
            messages[0] = message;

            requestBody.put("messages", messages);
            requestBody.put("temperature", temperature != null ? temperature : 0.7);
            requestBody.put("max_tokens", maxTokens != null ? maxTokens : 2000);

            // 发送HTTP请求
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            // 构建完整的API URL
            String apiUrl = baseUrl;
            if (!apiUrl.endsWith("/chat/completions")) {
                if (!apiUrl.endsWith("/")) {
                    apiUrl += "/";
                }
                apiUrl += "chat/completions";
            }

            ResponseEntity<String> response = restTemplate.exchange(
                    apiUrl, HttpMethod.POST, entity, String.class);

            if (response.getStatusCode().is2xxSuccessful()) {
                JSONObject responseJson = JSONUtil.parseObj(response.getBody());
                JSONArray choices = responseJson.getJSONArray("choices");
                if (choices != null && choices.size() > 0) {
                    JSONObject choice = choices.getJSONObject(0);
                    JSONObject messageObj = choice.getJSONObject("message");
                    return messageObj.getStr("content");
                }
            } else {
                log.error("LLM API调用失败，状态码：{}，响应：{}", response.getStatusCode(), response.getBody());
            }
        } catch (Exception e) {
            log.error("调用LLM服务生成总结时发生异常，modelId: {}", modelId, e);
        }

        return "生成总结失败，请稍后重试";
    }

    @Override
    public String generateSummary(String conversation, String promptTemplate) {
        return generateSummary(conversation, promptTemplate, null);
    }

    @Override
    public String generateSummaryWithHistory(String conversation, String historyMemory, String promptTemplate,
            String modelId) {
        if (!isAvailable()) {
            log.warn("LLM service unavailable, unable to generate summary.");
            return "LLM service unavailable, unable to generate summary.";
        }

        try {
            // 从智控台获取LLM模型配置
            ModelConfigEntity llmConfig;
            if (modelId != null && !modelId.trim().isEmpty()) {
                // 通过具体模型ID获取配置
                llmConfig = modelConfigService.getModelByIdFromCache(modelId);
            } else {
                // 保持向后兼容，使用默认配置
                llmConfig = getDefaultLLMConfig();
            }

            if (llmConfig == null || llmConfig.getConfigJson() == null) {
                log.error("No available LLM model configuration found，modelId: {}", modelId);
                return "No available LLM model configuration found";
            }

            JSONObject configJson = llmConfig.getConfigJson();
            String baseUrl = configJson.getStr("base_url");
            String model = configJson.getStr("model_name");
            String apiKey = configJson.getStr("api_key");

            if (StringUtils.isBlank(baseUrl) || StringUtils.isBlank(apiKey)) {
                log.error("LLM配置不完整，baseUrl或apiKey为空");
                return "The LLM configuration is incomplete, and a summary cannot be generated.";
            }

            // 构建提示词，包含历史记忆
            String prompt = (promptTemplate != null ? promptTemplate : DEFAULT_SUMMARY_PROMPT)
                    .replace("{history_memory}", historyMemory != null ? historyMemory : "No historical memory")
                    .replace("{conversation}", conversation);

            // 构建请求体
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("model", model != null ? model : "gpt-3.5-turbo");

            Map<String, Object>[] messages = new Map[1];
            Map<String, Object> message = new HashMap<>();
            message.put("role", "user");
            message.put("content", prompt);
            messages[0] = message;

            requestBody.put("messages", messages);
            requestBody.put("temperature", 0.2);
            requestBody.put("max_tokens", 2000);

            // 发送HTTP请求
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            headers.set("Authorization", "Bearer " + apiKey);

            HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

            // 构建完整的API URL
            String apiUrl = baseUrl;
            if (!apiUrl.endsWith("/chat/completions")) {
                if (!apiUrl.endsWith("/")) {
                    apiUrl += "/";
                }
                apiUrl += "chat/completions";
            }

            ResponseEntity<String> response = restTemplate.exchange(
                    apiUrl, HttpMethod.POST, entity, String.class);

            if (response.getStatusCode().is2xxSuccessful()) {
                JSONObject responseJson = JSONUtil.parseObj(response.getBody());
                JSONArray choices = responseJson.getJSONArray("choices");
                if (choices != null && choices.size() > 0) {
                    JSONObject choice = choices.getJSONObject(0);
                    JSONObject messageObj = choice.getJSONObject("message");
                    return messageObj.getStr("content");
                }
            } else {
                log.error("LLM API调用失败，状态码：{}，响应：{}", response.getStatusCode(), response.getBody());
            }
        } catch (Exception e) {
            log.error("An exception occurred while calling the LLM service to generate a summary，modelId: {}", modelId, e);
        }

        return "Summary generation failed, please try again later.";
    }

    @Override
    public boolean isAvailable() {
        try {
            ModelConfigEntity defaultLLMConfig = getDefaultLLMConfig();
            if (defaultLLMConfig == null || defaultLLMConfig.getConfigJson() == null) {
                return false;
            }

            JSONObject configJson = defaultLLMConfig.getConfigJson();
            String baseUrl = configJson.getStr("base_url");
            String apiKey = configJson.getStr("api_key");

            return baseUrl != null && !baseUrl.trim().isEmpty() &&
                    apiKey != null && !apiKey.trim().isEmpty();
        } catch (Exception e) {
            log.error("An exception occurred while checking the availability of the LLM service：", e);
            return false;
        }
    }

    @Override
    public boolean isAvailable(String modelId) {
        try {
            if (modelId == null || modelId.trim().isEmpty()) {
                return isAvailable();
            }

            // 通过具体模型ID获取配置
            ModelConfigEntity modelConfig = modelConfigService.getModelByIdFromCache(modelId);
            if (modelConfig == null || modelConfig.getConfigJson() == null) {
                log.warn("未找到指定的LLM模型配置，modelId: {}", modelId);
                return false;
            }

            JSONObject configJson = modelConfig.getConfigJson();
            String baseUrl = configJson.getStr("base_url");
            String apiKey = configJson.getStr("api_key");

            return baseUrl != null && !baseUrl.trim().isEmpty() &&
                    apiKey != null && !apiKey.trim().isEmpty();
        } catch (Exception e) {
            log.error("检查LLM服务可用性时发生异常，modelId: {}", modelId, e);
            return false;
        }
    }

    /**
     * 从智控台获取默认的LLM模型配置
     */
    private ModelConfigEntity getDefaultLLMConfig() {
        try {
            // 获取所有启用的LLM模型配置
            List<ModelConfigEntity> llmConfigs = modelConfigService.getEnabledModelsByType("LLM");
            if (llmConfigs == null || llmConfigs.isEmpty()) {
                return null;
            }

            // 优先返回默认配置，如果没有默认配置则返回第一个启用的配置
            for (ModelConfigEntity config : llmConfigs) {
                if (config.getIsDefault() != null && config.getIsDefault() == 1) {
                    return config;
                }
            }

            return llmConfigs.get(0);
        } catch (Exception e) {
            log.error("获取LLM模型配置时发生异常：", e);
            return null;
        }
    }
}