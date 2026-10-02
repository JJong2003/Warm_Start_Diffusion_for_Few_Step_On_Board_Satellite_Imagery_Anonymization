package io.github.JJong2003.model_showcase.ratelimit;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Deque;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.TimeUnit;

/**
 * IP당 요청 속도 제한. /api/** 의 작업 제출(POST)에만 적용된다.
 * 작업 상태 조회(GET /api/jobs/{id})는 대상에서 뺀다 — 프런트엔드가 0.7초 간격으로
 * 폴링하는데, 처리 시간이 길어지면(특히 CPU 추론) 폴링만으로 분당 한도를 다 써버려서
 * 아직 끝나지도 않은 자기 작업이 429로 막히는 문제가 있었다. 남용 방지가 목적인
 * 제출 요청만 제한하면 충분하다.
 */
@Component
public class RateLimitFilter extends OncePerRequestFilter {

    private final int requestsPerMinute;
    private final Map<String, Deque<Long>> hits = new ConcurrentHashMap<>();

    public RateLimitFilter(@Value("${app.rate-limit.requests-per-minute:30}") int requestsPerMinute) {
        this.requestsPerMinute = requestsPerMinute;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
            throws ServletException, IOException {
        boolean isRateLimited = request.getRequestURI().startsWith("/api/")
                && !"GET".equalsIgnoreCase(request.getMethod());
        if (!isRateLimited) {
            chain.doFilter(request, response);
            return;
        }

        String clientIp = extractClientIp(request);
        long now = System.currentTimeMillis();
        Deque<Long> timestamps = hits.computeIfAbsent(clientIp, k -> new ConcurrentLinkedDeque<>());

        synchronized (timestamps) {
            while (!timestamps.isEmpty() && now - timestamps.peekFirst() > TimeUnit.MINUTES.toMillis(1)) {
                timestamps.pollFirst();
            }
            if (timestamps.size() >= requestsPerMinute) {
                response.setStatus(429);
                response.setContentType("application/json;charset=UTF-8");
                response.getWriter().write(
                        "{\"error\":\"too_many_requests\",\"message\":\"요청이 너무 많습니다. 잠시 후 다시 시도해주세요.\"}");
                return;
            }
            timestamps.addLast(now);
        }

        chain.doFilter(request, response);
    }

    private String extractClientIp(HttpServletRequest request) {
        String forwardedFor = request.getHeader("X-Forwarded-For");
        if (forwardedFor != null && !forwardedFor.isBlank()) {
            return forwardedFor.split(",")[0].trim();
        }
        return request.getRemoteAddr();
    }
}
