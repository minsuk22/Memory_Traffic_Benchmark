package com.memtraffic.bench;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

/**
 * Minimal UI around the native benchmark. Arguments are the same as the CLI
 * (see `membench --help`). Output is also written to logcat with tag "membench".
 * Launch with arguments from adb:
 *   adb shell am start -n com.memtraffic.bench/.MainActivity --es args "--size-mb 1024 --sweep"
 * Without adb (e.g. devices managed by an MDM), "Peak finder" / "CPU random" / "Rand vs DVFS" run --peak / --cpu / --dvfs;
 * "Share log" / "Copy log" hand the complete output to any app.
 */
public class MainActivity extends Activity {
    static { System.loadLibrary("membench_jni"); }

    private native int nativeRun(String args);

    private TextView out;
    private ScrollView scroll;
    private Button run, peak, cpu, dvfs, share, copy;
    private EditText args;
    private final StringBuilder logText = new StringBuilder();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        // Keep the screen on so the SoC is not put into a low-power state mid-run.
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(24, 24, 24, 24);

        args = new EditText(this);
        args.setSingleLine(true);
        String extra = getIntent().getStringExtra("args");
        args.setText(extra != null ? extra : "--size-mb 1024");
        root.addView(args);

        LinearLayout row1 = new LinearLayout(this);
        run = button(row1, "Run benchmark", v -> start(args.getText().toString()));
        peak = button(row1, "Peak finder", v -> start(args.getText().toString() + " --peak"));
        cpu = button(row1, "CPU random", v -> start(args.getText().toString() + " --cpu"));
        root.addView(row1);
        LinearLayout row2 = new LinearLayout(this);
        dvfs = button(row2, "Rand vs DVFS", v -> start(args.getText().toString() + " --dvfs"));
        share = button(row2, "Share log", v -> shareLog());
        copy = button(row2, "Copy log", v -> copyLog());
        root.addView(row2);

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

    private Button button(LinearLayout row, String text, android.view.View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(text);
        b.setAllCaps(false);
        b.setOnClickListener(l);
        row.addView(b, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));
        return b;
    }

    private void start(final String a) {
        run.setEnabled(false);
        peak.setEnabled(false);
        cpu.setEnabled(false);
        dvfs.setEnabled(false);
        out.setText("");
        logText.setLength(0);
        logText.append("device: ").append(Build.MANUFACTURER).append(' ').append(Build.MODEL)
               .append(" (").append(Build.HARDWARE).append("), Android ").append(Build.VERSION.RELEASE)
               .append("\nargs: ").append(a.trim()).append("\n");
        out.append(logText);
        new Thread(() -> {
            int rc = nativeRun(a);
            runOnUiThread(() -> {
                onLogUi("exit code " + rc);
                run.setEnabled(true);
                peak.setEnabled(true);
                cpu.setEnabled(true);
                dvfs.setEnabled(true);
            });
        }, "membench").start();
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
