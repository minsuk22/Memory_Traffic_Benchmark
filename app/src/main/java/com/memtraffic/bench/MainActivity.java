package com.memtraffic.bench;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

/**
 * Minimal UI around the native benchmark. Arguments are the same as the CLI
 * (see `membench --help`). Output is also written to logcat with tag "membench".
 * Launch with arguments from adb:
 *   adb shell am start -n com.memtraffic.bench/.MainActivity --es args "--size-mb 1024 --steps"
 * Menu, top to bottom: random read with the access pattern constrained step by step
 * (--steps, or one step with --step N), then the other measurements (normal benchmark,
 * --rand-tune, and the diagnosis modes --peak / --cpu / --dvfs / --matrix under "Advanced").
 * The last line of every log is the random read result. "Share log" / "Copy log" hand the
 * complete output to any app; every log is also saved to Download/membench/.
 */
public class MainActivity extends Activity {
    static { System.loadLibrary("membench_jni"); }

    private native int nativeRun(String args);

    private TextView out, status;
    private ScrollView scroll;
    private EditText args;
    private final List<Button> runButtons = new ArrayList<>();
    private final StringBuilder logText = new StringBuilder();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        // Keep the screen on so the SoC is not put into a low-power state mid-run.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 24, 24, 24);

        title(root, "Random read: 접근 범위를 한 단계씩 좁히기");
        LinearLayout r0 = row(root);
        runButton(r0, "전체 단계 1 → 7 실행 (약 1분)", "--steps");
        LinearLayout r1 = row(root);
        runButton(r1, "1\n제한 없음", "--step 1");
        runButton(r1, "2\n64 MiB", "--step 2");
        runButton(r1, "3\n16 MiB", "--step 3");
        runButton(r1, "4\n4 MiB", "--step 4");
        LinearLayout r2 = row(root);
        runButton(r2, "5\n1 MiB", "--step 5");
        runButton(r2, "6\n1 MiB 동시", "--step 6");
        runButton(r2, "7\n+ 커널 튜닝", "--step 7");

        title(root, "기타 측정");
        LinearLayout r3 = row(root);
        runButton(r3, "기본 측정\nread/write/copy", "");
        runButton(r3, "자동 튜닝\n(rand-tune)", "--rand-tune");
        final LinearLayout adv = new LinearLayout(this);
        button(r3, "고급 진단 ▼", v -> adv.setVisibility(adv.getVisibility() == View.GONE ? View.VISIBLE : View.GONE));
        runButton(adv, "Peak", "--peak");
        runButton(adv, "CPU", "--cpu");
        runButton(adv, "DVFS", "--dvfs");
        runButton(adv, "Matrix", "--matrix");
        adv.setVisibility(View.GONE);
        root.addView(adv);

        LinearLayout r4 = row(root);
        args = new EditText(this);
        args.setSingleLine(true);
        args.setTextSize(12f);
        args.setHint("추가 옵션 (CLI와 동일)");
        String extra = getIntent().getStringExtra("args");
        args.setText("--size-mb 1024");
        r4.addView(args, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2f));
        button(r4, "로그 공유", v -> shareLog());
        button(r4, "로그 복사", v -> copyLog());

        status = new TextView(this);
        status.setTextSize(11f);
        root.addView(status);

        scroll = new ScrollView(this);
        out = new TextView(this);
        out.setTypeface(Typeface.MONOSPACE);
        out.setTextSize(9f);
        out.setTextIsSelectable(true);
        scroll.addView(out);
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);

        if (extra != null) start(extra);  // started from adb with --es args: run immediately
    }

    private void title(LinearLayout root, String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        t.setPadding(0, 16, 0, 4);
        root.addView(t);
    }

    private LinearLayout row(LinearLayout root) {
        LinearLayout r = new LinearLayout(this);
        root.addView(r);
        return r;
    }

    private Button button(LinearLayout row, String text, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setTextSize(12f);
        b.setOnClickListener(l);
        row.addView(b, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        return b;
    }

    /** A button that runs the benchmark with the option field plus `mode`. */
    private void runButton(LinearLayout row, String text, String mode) {
        runButtons.add(button(row, text, v -> start(args.getText().toString() + " " + mode)));
    }

    private void setRunning(boolean running) {
        for (Button b : runButtons) b.setEnabled(!running);
    }

    private void start(final String a) {
        setRunning(true);
        status.setText("측정 중... (" + a.trim() + ")");
        out.setText("");
        logText.setLength(0);
        logText.append("device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
               .append(" (").append(Build.HARDWARE).append("), Android ").append(Build.VERSION.RELEASE)
               .append("\nargs: ").append(a.trim()).append("\n");
        out.append(logText);
        new Thread(() -> {
            int rc = nativeRun(a);
            runOnUiThread(() -> {
                // Kept out of the log so its last line stays the random read result.
                status.setText((rc == 0 ? "완료" : rc == 1 ? "완료 - 실패한 검사 있음" : "오류")
                        + " (exit " + rc + "), 로그 저장: " + saveLog());
                setRunning(false);
            });
        }, "membench").start();
    }

    /** Writes the log to Download/membench/ (visible over USB file transfer, no permission needed). */
    private String saveLog() {
        String name = "membench_" + new java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US)
                .format(new java.util.Date()) + ".txt";
        byte[] data = logText.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
        try {
            if (Build.VERSION.SDK_INT >= 29) {
                android.content.ContentValues v = new android.content.ContentValues();
                v.put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, name);
                v.put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "text/plain");
                v.put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, "Download/membench");
                android.net.Uri uri = getContentResolver().insert(
                        android.provider.MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
                if (uri == null) return "not saved (MediaStore insert failed)";
                try (java.io.OutputStream os = getContentResolver().openOutputStream(uri)) { os.write(data); }
                return "Download/membench/" + name;
            }
            java.io.File f = new java.io.File(getExternalFilesDir(null), name);
            try (java.io.FileOutputStream os = new java.io.FileOutputStream(f)) { os.write(data); }
            return f.getAbsolutePath();
        } catch (Exception e) {
            return "not saved (" + e + ")";
        }
    }

    private void shareLog() {
        Intent i = new Intent(Intent.ACTION_SEND);
        i.setType("text/plain");
        i.putExtra(Intent.EXTRA_SUBJECT, "membench log " + Build.MODEL);
        i.putExtra(Intent.EXTRA_TEXT, logText.toString());
        startActivity(Intent.createChooser(i, "Share membench log"));
    }

    private void copyLog() {
        ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        cm.setPrimaryClip(ClipData.newPlainText("membench log", logText.toString()));
        Toast.makeText(this, "Log copied (" + logText.length() + " characters)", Toast.LENGTH_SHORT).show();
    }

    /** Called from native code on the benchmark thread. */
    @SuppressWarnings("unused")
    private void onLog(String line) {
        runOnUiThread(() -> onLogUi(line));
    }

    private void onLogUi(String line) {
        logText.append(line).append('\n');
        out.append(line);
        out.append("\n");
        scroll.post(() -> scroll.fullScroll(ScrollView.FOCUS_DOWN));
    }
}
