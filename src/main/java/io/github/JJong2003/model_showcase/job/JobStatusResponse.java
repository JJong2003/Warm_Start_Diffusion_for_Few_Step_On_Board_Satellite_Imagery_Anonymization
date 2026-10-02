package io.github.JJong2003.model_showcase.job;

public record JobStatusResponse(
        JobStatus status,
        Integer queuePosition,
        PipelineResult result,
        String errorMessage
) {
}
