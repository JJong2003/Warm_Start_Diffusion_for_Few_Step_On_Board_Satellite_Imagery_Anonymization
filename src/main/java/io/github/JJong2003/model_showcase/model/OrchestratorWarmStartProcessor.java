package io.github.JJong2003.model_showcase.model;

import io.github.JJong2003.model_showcase.job.PipelineResult;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.http.converter.json.JacksonJsonHttpMessageConverter;
import org.springframework.stereotype.Component;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.json.JsonMapper;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

/**
 * Python GPU 오케스트레이터(orchestrator/server.py)를 호출해 실제 논문 파이프라인
 * (YOLO11s-OBB + SD1.5 LoRA Inpainting) 결과를 받아오는 프로세서.
 */
@Component
public class OrchestratorWarmStartProcessor implements ModelProcessor {

    private final RestClient restClient;

    public OrchestratorWarmStartProcessor(JsonMapper jacksonJsonMapper,
                                           @Value("${app.orchestrator.base-url}") String baseUrl,
                                           @Value("${app.orchestrator.api-key:}") String apiKey) {
        // application.yaml의 property-naming-strategy: SNAKE_CASE가 반영된 Spring Boot의
        // JsonMapper 빈을 그대로 써야 한다. RestClient.builder()(static factory)로 새로 만들면
        // 별도의 기본 설정 JsonMapper를 써서 오케스트레이터가 보내는 snake_case 응답
        // (boxes_image 등)을 PipelineResult의 camelCase 필드에 못 붙인다.
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(10_000);
        factory.setReadTimeout(600_000); // CPU 추론은 수 분 걸릴 수 있음
        RestClient.Builder builder = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(factory)
                .configureMessageConverters(clientBuilder ->
                        clientBuilder.withJsonConverter(new JacksonJsonHttpMessageConverter(jacksonJsonMapper)));
        if (!apiKey.isBlank()) {
            builder.defaultHeader("X-Api-Key", apiKey); // Modal 엔드포인트 보호용 공유 키
        }
        this.restClient = builder.build();
    }

    @Override
    public PipelineResult process(BufferedImage input) throws IOException {
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", new ByteArrayResource(toPng(input)) {
            @Override
            public String getFilename() {
                return "input.png";
            }
        });
        body.add("model_id", "warm-start-anonymizer");

        return restClient.post()
                .uri("/infer")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(PipelineResult.class);
    }

    private byte[] toPng(BufferedImage image) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        return baos.toByteArray();
    }
}
