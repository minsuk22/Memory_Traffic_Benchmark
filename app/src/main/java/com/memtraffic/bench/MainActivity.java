package com.memtraffic.bench;

import android.app.Activity;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * Minimal UI around the native benchmark. Arguments are the same as the CLI
 * (see `membench --help`). Output is also written to logcat with tag "membench".
 * Launch with arguments from adb:
 *   adb shell am start -n com.memtraffic.bench/.MainActivity --es args "--size-mb 1024 --sweep"
 */
public class MainActivity extends Activity {
    static { System.loadLibrary("membench_jni"); }

    private native int nativeRun(String args);

    private TextView out;
    private ScrollView scroll;
    private Button run;
    private EditText args;

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

        run = new Button(this);
        run.setText("Run benchmark");
        run.setOnClickListener(v -> start());
        root.addView(run);

        scroll = new ScrollView(this);
        out = new TextView(this);
        out.setTypeface(Typeface.MONOSPACE);
        out.setTextSize(9f);
        out.setTextIsSelectable(true);
        scroll.addView(out);
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);

        if (extra != null) start();  // started from adb with --es args: run immediately
    }

    private void start() {
        run.setEnabled(false);
        out.setText("");
        final String a = args.getText().toString();
        new Thread(() -> {
            int rc = nativeRun(a);
            runOnUiThread(() -> {
                onLogUi("exit code " + rc);
                run.setEnabled(true);
            });
        }, "membench").start();
    }

    /** Called from native code on the benchmark thread. */
    @SuppressWarnings("unused")
    private void onLog(String line) {
        runOnUiThread(() -> onLogUi(line));
    }

    private void onLogUi(String line) {
        out.append(line);
        out.append("\n");
        scroll.post(() -> scroll.fullScroll(ScrollView.FOCUS_DOWN));
    }
}
