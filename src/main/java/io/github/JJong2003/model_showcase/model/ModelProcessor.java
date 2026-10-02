package io.github.JJong2003.model_showcase.model;

import io.github.JJong2003.model_showcase.job.PipelineResult;

import java.awt.image.BufferedImage;
import java.io.IOException;

public interface ModelProcessor {
    PipelineResult process(BufferedImage input) throws IOException;
}
