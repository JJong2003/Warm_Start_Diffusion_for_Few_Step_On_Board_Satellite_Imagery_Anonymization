package io.github.JJong2003.model_showcase.job;

import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Locale;

public enum JobStatus {
    QUEUED, PROCESSING, DONE, ERROR;

    @JsonValue
    public String toJson() {
        return name().toLowerCase(Locale.ROOT);
    }
}
