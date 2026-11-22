package vn.vjaa.vnindexvol;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Bundle;
import android.view.Window;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * VN-Index · Dự báo biến động (ứng dụng minh hoạ luận văn).
 * Có mạng: mở trang GitHub Pages (giao diện và dữ liệu mới nhất, cập nhật mỗi phiên).
 * Không có mạng hoặc lỗi tải: mở bản giao diện đóng gói sẵn trong APK, dùng dữ liệu đã lưu lần trước.
 */
public class MainActivity extends Activity {

    private static final String LOCAL = "file:///android_asset/index.html";
    private WebView web;
    private String remote;
    private boolean fellBack = false;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setStatusBarColor(Color.parseColor("#0E1B2C"));
        getWindow().setNavigationBarColor(Color.parseColor("#0E1B2C"));

        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#0E1B2C"));
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);       // localStorage: lưu dữ liệu dùng ngoại tuyến
        s.setAllowFileAccess(true);
        s.setAllowFileAccessFromFileURLs(true);   // bản ngoại tuyến đọc dữ liệu JSON đóng gói
        s.setCacheMode(WebSettings.LOAD_DEFAULT);
        s.setSupportZoom(false);
        s.setUseWideViewPort(true);
        s.setLoadWithOverviewMode(true);
        s.setUserAgentString(s.getUserAgentString() + " VNIndexVolApp");

        web.setWebChromeClient(new WebChromeClient());
        web.setWebViewClient(new WebViewClient() {
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                return openExternal(request.getUrl());
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                if (request.isForMainFrame() && !fellBack) {
                    fellBack = true;
                    view.loadUrl(LOCAL);
                }
            }
        });
        setContentView(web);

        remote = readAsset("remote.txt");
        if (savedInstanceState != null) {
            web.restoreState(savedInstanceState);
        } else if (remote != null && !remote.isEmpty() && online()) {
            web.loadUrl(remote);
        } else {
            fellBack = true;
            web.loadUrl(LOCAL);
        }
    }

    /** Liên kết ra ngoài trang ứng dụng thì mở bằng trình duyệt của máy. */
    private boolean openExternal(Uri uri) {
        String u = uri.toString();
        if (u.startsWith("file:") || (remote != null && !remote.isEmpty() && u.startsWith(remote))) {
            return false;
        }
        if ("http".equals(uri.getScheme()) || "https".equals(uri.getScheme())) {
            try {
                startActivity(new Intent(Intent.ACTION_VIEW, uri));
            } catch (Exception ignored) {
            }
            return true;
        }
        return false;
    }

    @SuppressWarnings("deprecation")
    private boolean online() {
        try {
            ConnectivityManager cm = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
            NetworkInfo ni = cm.getActiveNetworkInfo();
            return ni != null && ni.isConnected();
        } catch (Exception e) {
            return true;
        }
    }

    private String readAsset(String name) {
        try (BufferedReader r = new BufferedReader(new InputStreamReader(getAssets().open(name), "UTF-8"))) {
            String line = r.readLine();
            return line == null ? null : line.trim();
        } catch (Exception e) {
            return null;
        }
    }

    @SuppressWarnings("deprecation")
    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (web != null) web.saveState(outState);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (web != null) web.onPause();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (web != null) web.onResume();
    }

    @Override
    protected void onDestroy() {
        if (web != null) {
            web.destroy();
            web = null;
        }
        super.onDestroy();
    }
}
