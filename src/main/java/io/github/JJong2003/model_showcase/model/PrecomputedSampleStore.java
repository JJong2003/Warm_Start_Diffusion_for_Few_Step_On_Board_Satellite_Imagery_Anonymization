package io.github.JJong2003.model_showcase.model;

import io.github.JJong2003.model_showcase.job.PipelineResult;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.Optional;

/**
 * orchestrator/precompute_samples.py로 미리 뽑아둔 샘플 갤러리 결과
 * (src/main/resources/precomputed/samples.json)를 부팅 시 메모리에 올려둔다.
 * 샘플 이미지는 이 스토어에서 즉시 응답하고, GPU 큐/오케스트레이터를 거치지 않는다.
 */
@Component
public class PrecomputedSampleStore {

    private final Map<String, PipelineResult> samples;

    public PrecomputedSampleStore(JsonMapper jacksonJsonMapper) throws IOException {
        ClassPathResource resource = new ClassPathResource("precomputed/samples.json");
        try (InputStream in = resource.getInputStream()) {
            this.samples = jacksonJsonMapper.readValue(in, new TypeReference<Map<String, PipelineResult>>() {});
        }
    }

    public Optional<PipelineResult> find(String sampleId) {
        return Optional.ofNullable(samples.get(sampleId));
    }
}
