package io.github.JJong2003.model_showcase.model;

import io.github.JJong2003.model_showcase.job.PipelineMeta;
import io.github.JJong2003.model_showcase.job.PipelineResult;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.util.Random;

/**
 * 실제 논문 파이프라인(YOLO11s-OBB + SD1.5 LoRA Inpainting)은 Python GPU
 * 오케스트레이터가 아직 준비되지 않아, 그때까지 API 계약(5단계 이미지 + meta)을
 * 실제로 만족시키는 자리표시(placeholder) 처리기. 프런트엔드 데모 모드가 클라이언트
 * 캔버스로 흉내내던 것과 같은 시각적 논리를 서버 쪽에서 재현한다.
 */
@Component
public class WarmStartDemoProcessor implements ModelProcessor {

    private static final Color BOX_COLOR = new Color(0xE8, 0x66, 0x4F);
    private static final Random RANDOM = new Random();

    @Override
    public PipelineResult process(BufferedImage input) throws IOException {
        long start = System.currentTimeMillis();

        BufferedImage original = toArgb(input);
        int width = original.getWidth();
        int height = original.getHeight();

        double cx = width * (0.32 + RANDOM.nextDouble() * 0.28);
        double cy = height * (0.32 + RANDOM.nextDouble() * 0.28);
        double boxWidth = width * 0.22;
        double boxHeight = height * 0.14;
        double angle = Math.toRadians(RANDOM.nextDouble() * 30 - 15);

        BufferedImage boxesImage = drawDetectionBox(original, cx, cy, boxWidth, boxHeight, angle);
        BufferedImage maskImage = createMask(width, height, cx, cy, boxWidth, boxHeight, angle);
        BufferedImage fillImage = applyClippedBlur(original, cx, cy, boxWidth, boxHeight, angle, 6);
        BufferedImage outputImage = applyClippedBlur(original, cx, cy, boxWidth, boxHeight, angle, 2);

        long latencyMs = System.currentTimeMillis() - start;

        return new PipelineResult(
                toDataUrl(boxesImage),
                toDataUrl(maskImage),
                toDataUrl(fillImage),
                toDataUrl(outputImage),
                toDataUrl(outputImage),
                new PipelineMeta(1, 10, latencyMs, true)
        );
    }

    private BufferedImage drawDetectionBox(BufferedImage source, double cx, double cy,
                                            double bw, double bh, double angle) {
        BufferedImage out = deepCopy(source);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.translate(cx, cy);
        g.rotate(angle);
        g.setColor(BOX_COLOR);
        float strokeWidth = (float) Math.max(2, out.getWidth() * 0.006);
        g.setStroke(new BasicStroke(strokeWidth, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10, new float[]{8, 6}, 0));
        g.drawRect((int) (-bw / 2), (int) (-bh / 2), (int) bw, (int) bh);
        g.setTransform(new AffineTransform());
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, (int) Math.max(12, out.getWidth() * 0.02)));
        g.drawString("object 0.9" + RANDOM.nextInt(9), (float) (cx - bw / 2), (float) (cy - bh / 2 - 6));
        g.dispose();
        return out;
    }

    private BufferedImage createMask(int width, int height, double cx, double cy,
                                      double bw, double bh, double angle) {
        BufferedImage mask = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = mask.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, width, height);
        g.translate(cx, cy);
        g.rotate(angle);
        g.setColor(Color.WHITE);
        g.fill(new Rectangle2D.Double(-bw * 0.58, -bh * 0.62, bw * 1.16, bh * 1.24));
        g.dispose();
        return mask;
    }

    private BufferedImage applyClippedBlur(BufferedImage source, double cx, double cy,
                                            double bw, double bh, double angle, int blurPasses) {
        BufferedImage out = deepCopy(source);

        AffineTransform transform = new AffineTransform();
        transform.translate(cx, cy);
        transform.rotate(angle);
        Rectangle2D localRect = new Rectangle2D.Double(-bw * 0.58, -bh * 0.62, bw * 1.16, bh * 1.24);
        Shape rotatedShape = transform.createTransformedShape(localRect);
        Rectangle bounds = rotatedShape.getBounds();

        int pad = 12;
        int bx = Math.max(0, bounds.x - pad);
        int by = Math.max(0, bounds.y - pad);
        int bw2 = Math.min(out.getWidth() - bx, bounds.width + pad * 2);
        int bh2 = Math.min(out.getHeight() - by, bounds.height + pad * 2);
        if (bw2 <= 0 || bh2 <= 0) {
            return out;
        }

        BufferedImage sub = deepCopy(out.getSubimage(bx, by, bw2, bh2));
        BufferedImage blurred = sub;
        for (int i = 0; i < blurPasses; i++) {
            blurred = boxBlurPass(blurred);
        }

        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setClip(rotatedShape);
        g.drawImage(blurred, bx, by, null);
        g.dispose();
        return out;
    }

    private BufferedImage boxBlurPass(BufferedImage src) {
        float[] kernelData = new float[25];
        java.util.Arrays.fill(kernelData, 1f / 25f);
        Kernel kernel = new Kernel(5, 5, kernelData);
        ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_ARGB);
        op.filter(src, dst);
        return dst;
    }

    private BufferedImage deepCopy(BufferedImage bi) {
        BufferedImage copy = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = copy.createGraphics();
        g.drawImage(bi, 0, 0, null);
        g.dispose();
        return copy;
    }

    private BufferedImage toArgb(BufferedImage bi) {
        if (bi.getType() == BufferedImage.TYPE_INT_ARGB) {
            return bi;
        }
        return deepCopy(bi);
    }

    private String toDataUrl(BufferedImage image) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(baos.toByteArray());
    }
}
