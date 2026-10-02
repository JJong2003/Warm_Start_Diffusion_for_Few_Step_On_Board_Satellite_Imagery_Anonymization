package io.github.JJong2003.model_showcase.job;

import io.github.JJong2003.model_showcase.model.ModelProcessor;
import io.github.JJong2003.model_showcase.model.PrecomputedSampleStore;
import jakarta.annotation.PostConstruct;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 동아리 공유 GPU를 흉내내는 전역 FIFO 큐. 어느 조 모델이든 model_id와 무관하게
 * 같은 줄에 서고, 워커 스레드 하나가 순서대로 처리한다 (모델별 큐가 아님).
 */
@Service
public class JobQueueService {

    private final Map<String, ModelProcessor> modelRegistry;
    private final PrecomputedSampleStore precomputedSampleStore;
    private final Map<String, Job> jobs = new ConcurrentHashMap<>();
    private final BlockingQueue<Job> queue;

    public JobQueueService(Map<String, ModelProcessor> modelRegistry,
                            PrecomputedSampleStore precomputedSampleStore,
                            @Value("${app.queue.max-length:20}") int maxQueueLength) {
        this.modelRegistry = modelRegistry;
        this.precomputedSampleStore = precomputedSampleStore;
        this.queue = new LinkedBlockingQueue<>(maxQueueLength);
    }

    @PostConstruct
    void startWorker() {
        Thread worker = new Thread(this::runWorkerLoop, "job-worker");
        worker.setDaemon(true);
        worker.start();
    }

    private void runWorkerLoop() {
        while (true) {
            try {
                Job job = queue.take();
                job.setStatus(JobStatus.PROCESSING);
                try {
                    ModelProcessor processor = modelRegistry.get(job.getModelId());
                    if (processor == null) {
                        throw new IllegalStateException("알 수 없는 모델입니다: " + job.getModelId());
                    }
                    PipelineResult result = processor.process(job.getImage());
                    job.setResult(result);
                    job.setStatus(JobStatus.DONE);
                } catch (Exception e) {
                    job.setErrorMessage(e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
                    job.setStatus(JobStatus.ERROR);
                }
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    public Job submit(String modelId, java.awt.image.BufferedImage image) {
        if (!modelRegistry.containsKey(modelId)) {
            throw new IllegalArgumentException("알 수 없는 model_id 입니다: " + modelId);
        }
        Job job = new Job(UUID.randomUUID().toString(), modelId, image);
        jobs.put(job.getId(), job);
        boolean offered = queue.offer(job);
        if (!offered) {
            jobs.remove(job.getId());
            throw new QueueFullException("지금 대기열이 가득 찼습니다. 잠시 후 다시 시도해주세요.");
        }
        return job;
    }

    /**
     * 미리 계산해둔 샘플 결과를 즉시 반환하는 pseudo-job을 만든다. 실제 GPU 작업이 아니므로
     * 전역 큐(queue)에는 절대 넣지 않는다 — 넣으면 앞에 쌓인 무거운 작업 뒤에서 기다리게 되어
     * "가볍게 구경만 해보려는" 목적 자체가 무의미해진다.
     */
    public Job submitPrecomputed(String sampleId) {
        PipelineResult result = precomputedSampleStore.find(sampleId)
                .orElseThrow(() -> new IllegalArgumentException("알 수 없는 sample_id 입니다: " + sampleId));
        Job job = new Job(UUID.randomUUID().toString(), "warm-start-anonymizer", null);
        job.setResult(result);
        job.setStatus(JobStatus.DONE);
        jobs.put(job.getId(), job);
        return job;
    }

    public int queuePositionOf(Job job) {
        int position = 1;
        for (Job queued : queue) {
            if (queued.getId().equals(job.getId())) {
                return position;
            }
            position++;
        }
        return 0;
    }

    public Optional<Job> find(String jobId) {
        return Optional.ofNullable(jobs.get(jobId));
    }
}
