package io.github.JJong2003.model_showcase.model;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Map;

/**
 * 조별 모델 등록 레지스트리. model_id -> ModelProcessor.
 * 다른 조 모델을 추가할 때는 ModelProcessor 구현체를 만들고 이 맵에 등록하면 된다.
 *
 * app.orchestrator.enabled=false 로 두면 Python 오케스트레이터 없이도
 * WarmStartDemoProcessor(자리표시) 로 프론트엔드만 확인할 수 있다.
 */
@Configuration
public class ModelRegistryConfig {

    @Bean
    public Map<String, ModelProcessor> modelRegistry(
            @Value("${app.orchestrator.enabled:true}") boolean orchestratorEnabled,
            WarmStartDemoProcessor warmStartDemoProcessor,
            OrchestratorWarmStartProcessor orchestratorWarmStartProcessor) {
        ModelProcessor active = orchestratorEnabled ? orchestratorWarmStartProcessor : warmStartDemoProcessor;
        return Map.of("warm-start-anonymizer", active);
    }
}
