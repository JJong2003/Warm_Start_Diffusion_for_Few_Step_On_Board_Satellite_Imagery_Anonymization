package io.github.JJong2003.model_showcase.job;

import lombok.Getter;
import lombok.Setter;

import java.awt.image.BufferedImage;
import java.time.Instant;

@Getter
public class Job {

    private final String id;
    private final String modelId;
    private final BufferedImage image;
    private final Instant submittedAt = Instant.now();

    @Setter
    private volatile JobStatus status = JobStatus.QUEUED;
    @Setter
    private volatile PipelineResult result;
    @Setter
    private volatile String errorMessage;

    public Job(String id, String modelId, BufferedImage image) {
        this.id = id;
        this.modelId = modelId;
        this.image = image;
    }
}
