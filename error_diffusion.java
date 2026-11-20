import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicIntegerArray;
import javax.imageio.ImageIO;

class ErrorDiffusion
{
    static final int THRESHOLD = 128;               // pixel > 128 -> white, otherwise black
    static final String STUDENT_ID = "6734430223"; 

    static int W, H;                     // image width and height
    static int[] buf;                    // padded working copy (int so accumulated error can't overflow a byte)
    static byte[] outImg;                // 0 = black, 1 = white
    static AtomicIntegerArray progress;  // progress[i * PAD] = pixels finished in row i

   
    static int idx(int i, int j) { return i * (W + 2) + j + 1; }

   
    static void doPixel(int[] b, byte[] o, int i, int j)
    {
        int v = b[idx(i, j)];                 // current value (original + error received so far)
        int px = v > THRESHOLD ? 1 : 0;       // quantization: 1 = white, 0 = black
        o[i * W + j] = (byte) px;             
        int err = v - 255 * px;               // quantization error (how far off the rounding was)
        // Diffusion kernel: spread err to the 4 not-yet-processed neighbors
        //            [i][j]   7/16
        //   3/16      5/16    1/16
        b[idx(i,     j + 1)] += (err * 7) >> 4;   // right
        b[idx(i + 1, j - 1)] += (err * 3) >> 4;   // below-left
        b[idx(i + 1, j    )] += (err * 5) >> 4;   // below
        b[idx(i + 1, j + 1)] += (err * 1) >> 4;   // below-right
    }

    static final int BLOCK = 64;  // columns processed between progress updates (less sync traffic)
    static final int PAD = 16;    // 16 ints = 64 bytes: each row's counter on its own cache line
                                  // (prevents "false sharing" between threads' counters)

    // Code run by each worker thread.
    // tid = this thread's number (0..nt-1), nt = total number of threads.
    static void worker(int tid, int nt)
    {
        for (int i = tid; i < H; i += nt)          // rows interleaved across threads
        {                                          // (thread t does rows t, t+nt, t+2nt, ...)
            int seen = 0;                          // last progress value we read from row i-1
            for (int j0 = 0; j0 < W; j0 += BLOCK)  // go through the row in blocks of 64 columns
            {
                int j1 = Math.min(j0 + BLOCK, W);  // end of this block (exclusive)
                if (i > 0)                         // row 0 has no row above, so it never waits
                {
                    // row i-1 must have finished pixels 0..j1+1 (we touch up to column j1)
                    // i.e. stay about 2 pixels behind the row above (wavefront)
                    int need = Math.min(j1 + 2, W);
                    // pure busy-wait (no sleeping)
                    while (seen < need)
                    {
                        seen = progress.get((i - 1) * PAD);          // volatile read = acquire
                        if (seen < need) Thread.onSpinWait();        // CPU hint: we are spinning
                    }
                }
                for (int j = j0; j < j1; j++)      // process the pixels in this block
                    doPixel(buf, outImg, i, j);
                // release: error propagated BEFORE "done" is published
                progress.lazySet(i * PAD, j1);     // tell the next row: "row i is done up to column j1"
            }
        }
    }

    // Copy the grayscale image into the padded int buffer (dummy pixels stay 0).
    static void fill(int[] b, byte[] gray)
    {
        Arrays.fill(b, 0);
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                b[idx(i, j)] = gray[i * W + j] & 0xFF;   // & 0xFF: byte (signed) -> 0..255
    }

    // Single-thread version (baseline). Returns time in milliseconds.
    static double runSequential(byte[] gray, int size, byte[] result)
    {
        int[] sb = new int[size];
        fill(sb, gray);                              // pre-processing (not timed)
        long start = System.nanoTime();
        for (int i = 0; i < H; i++)                  // plain raster scan: left->right, top->bottom
            for (int j = 0; j < W; j++)
                doPixel(sb, result, i, j);
        long end = System.nanoTime();
        return (end - start) / 1e6;                  // ns -> ms
    }

    // Multi-thread version with n threads. Returns time in milliseconds.
    static double runParallel(byte[] gray, int size, int n) throws InterruptedException
    {
        buf = new int[size];
        outImg = new byte[W * H];
        progress = new AtomicIntegerArray(H * PAD);  // one counter per row, all start at 0
        fill(buf, gray);                             // pre-processing (not timed)

        Thread[] threads = new Thread[n];
        for (int t = 0; t < n; t++)
        {
            final int id = t;
            threads[t] = new Thread(() -> worker(id, n));
        }                                            // thread creation (not timed)

        long start = System.nanoTime();              // start timing after threads are created
        for (Thread t : threads) t.start();          // run all workers
        for (Thread t : threads) t.join();           // wait for all to finish
        long end = System.nanoTime();                // stop timing after join
        return (end - start) / 1e6;
    }

    public static void main(String[] args) throws Exception
    {
        String inputPath = "map-grayscale.png"; //input image path here naja

        if (inputPath.isEmpty() && args.length > 0) inputPath = args[0];   // 1st choice: command-line argument
        if (inputPath.isEmpty())                                           // 2nd choice: ask the user
        {
            System.out.print("Enter input image path: ");
            Scanner sc = new Scanner(System.in);
            String line = sc.hasNextLine() ? sc.nextLine() : "";
            inputPath = line.trim().replaceAll("^[\"']+|[\"']+$", "");     // remove surrounding quotes
        }
        File inFile = new File(inputPath);
        if (inputPath.isEmpty() || !inFile.isFile())                       // validate the file
        {
            System.out.println(inputPath.isEmpty() ? "No input image given." : "File not found: " + inputPath);
            System.exit(1);
        }
        // Output name = <input name>-error-diff-<student id>.png in the same folder
        String dir = inFile.getAbsoluteFile().getParent();
        String fileName = inFile.getName();
        int dot = fileName.lastIndexOf('.');
        String name = dot > 0 ? fileName.substring(0, dot) : fileName;
        File outFile = new File(dir, name + "-error-diff-" + STUDENT_ID + ".png");

        // ---------- 2) load the image and convert to 8-bit grayscale ----------
        // ---- pre-processing: load image -> 8-bit grayscale----
        BufferedImage img = ImageIO.read(inFile);
        if (img == null)
        {
            System.out.println("Cannot read image: " + inputPath);
            System.exit(1);
        }
        W = img.getWidth(); H = img.getHeight();
        byte[] gray = new byte[W * H];
        if (img.getType() == BufferedImage.TYPE_BYTE_GRAY)
        {
            // read raw gray values (getRGB would apply a gamma conversion and change them)
            img.getRaster().getDataElements(0, 0, W, H, gray);
        }
        else
        {
            // color image -> gray using the standard luminance weights
            for (int i = 0; i < H; i++)
                for (int j = 0; j < W; j++)
                {
                    int rgb = img.getRGB(j, i);
                    int r = (rgb >> 16) & 0xFF, g = (rgb >> 8) & 0xFF, b = rgb & 0xFF;
                    // same weights as PIL convert('L'); exact for an already-grayscale input
                    gray[i * W + j] = (byte) ((299 * r + 587 * g + 114 * b) / 1000);
                }
        }
        int size = (H + 1) * (W + 2);   // padded buffer: H rows + 1 dummy row, W columns + 2 dummy columns
        int cores = Runtime.getRuntime().availableProcessors();            // default: use all cores
        if (args.length > 1) cores = Math.max(1, Integer.parseInt(args[1]));   // optional thread count

        // warm-up so JIT compilation is not counted in t1 / t2 (Java's JIT needs several rounds)
        for (int k = 0; k < 5; k++)
        {
            runSequential(gray, size, new byte[W * H]);
            runParallel(gray, size, cores);
        }

        //t1: single core
        byte[] seqOut = new byte[W * H];
        double t1 = runSequential(gray, size, seqOut);

        //t2: multiple cores
        double t2 = runParallel(gray, size, cores);
        boolean ok = Arrays.equals(outImg, seqOut);   // correctness check: parallel must equal sequential

        //report
        System.out.println("Image: " + W + " x " + H + ", cores = " + cores);
        System.out.printf("t1 (single core)          = %.3f ms%n", t1);
        System.out.printf("%-26s= %.3f ms%n", "t2 (" + cores + " cores)", t2);
        System.out.printf("speedup t1/t2             = %.2fx%n", t1 / t2);   // speedup = sequential time / parallel time
        System.out.println("parallel result " + (ok ? "MATCHES" : "DIFFERS from") + " sequential");

        //save the black & white result
        BufferedImage res = new BufferedImage(W, H, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = res.getRaster();
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++)
                r.setSample(j, i, 0, outImg[i * W + j] == 1 ? 255 : 0);   // 1 -> 255 (white), 0 -> 0 (black)
        ImageIO.write(res, "png", outFile);
        System.out.println("Saved: " + outFile.getPath());
    }
}