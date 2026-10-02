package io.github.JJong2003.model_showcase.job;

public record PipelineResult(
        String boxesImage,
        String maskImage,
        String coarseFillImage,
        String diffusionPreviewImage,
        String resultImage,
        PipelineMeta meta
) {
}
