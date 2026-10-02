package io.github.JJong2003.model_showcase.job;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Set;

@RestController
@RequestMapping("/api/jobs")
public class JobController {

    private static final Set<String> ALLOWED_CONTENT_TYPES = Set.of("image/png", "image/jpeg");

    private final JobQueueService jobQueueService;
    private final int maxWidth;
    private final int maxHeight;

    public JobController(JobQueueService jobQueueService,
                          @Value("${app.upload.max-width:6000}") int maxWidth,
                          @Value("${app.upload.max-height:6000}") int maxHeight) {
        this.jobQueueService = jobQueueService;
        this.maxWidth = maxWidth;
        this.maxHeight = maxHeight;
    }

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<JobSubmitResponse> submit(
            @RequestParam(value = "image", required = false) MultipartFile image,
            @RequestParam(value = "sample_id", required = false) String sampleId,
            @RequestParam(value = "model_id", defaultValue = "warm-start-anonymizer") String modelId
    ) throws IOException {
        boolean hasImage = image != null && !image.isEmpty();
        boolean hasSampleId = sampleId != null && !sampleId.isBlank();
        if (hasImage == hasSampleId) {
            throw new IllegalArgumentException("image 파일 또는 sample_id 중 정확히 하나가 필요합니다.");
        }

        Job job = hasSampleId
                ? jobQueueService.submitPrecomputed(sampleId)
                : jobQueueService.submit(modelId, validateAndDecode(image));
        int position = jobQueueService.queuePositionOf(job);
        return ResponseEntity.status(HttpStatus.ACCEPTED)
                .body(new JobSubmitResponse(job.getId(), position));
    }

    @GetMapping("/{jobId}")
    public JobStatusResponse status(@PathVariable String jobId) {
        Job job = jobQueueService.find(jobId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "존재하지 않는 작업입니다."));

        return switch (job.getStatus()) {
            case QUEUED -> new JobStatusResponse(JobStatus.QUEUED, jobQueueService.queuePositionOf(job), null, null);
            case PROCESSING -> new JobStatusResponse(JobStatus.PROCESSING, null, null, null);
            case DONE -> new JobStatusResponse(JobStatus.DONE, null, job.getResult(), null);
            case ERROR -> new JobStatusResponse(JobStatus.ERROR, null, null, job.getErrorMessage());
        };
    }

    private BufferedImage validateAndDecode(MultipartFile image) throws IOException {
        if (image.isEmpty()) {
            throw new IllegalArgumentException("이미지 파일이 비어있습니다.");
        }
        String contentType = image.getContentType();
        if (contentType == null || !ALLOWED_CONTENT_TYPES.contains(contentType)) {
            throw new IllegalArgumentException("PNG 또는 JPG 이미지만 업로드할 수 있습니다.");
        }
        BufferedImage decoded = ImageIO.read(image.getInputStream());
        if (decoded == null) {
            throw new IllegalArgumentException("이미지를 읽을 수 없습니다.");
        }
        if (decoded.getWidth() > maxWidth || decoded.getHeight() > maxHeight) {
            throw new IllegalArgumentException(
                    "이미지 해상도가 너무 큽니다 (최대 " + maxWidth + "x" + maxHeight + ").");
        }
        return decoded;
    }
}
