import { useEffect, useMemo, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_BACKEND_URL || "").replace(/\/$/, "");

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

const mediaOptions = [
  {
    id: "image",
    title: "Image",
    accept: "image/*",
    endpoint: "/api/analyze/image",
    helper: "Upload a portrait, profile photo, or any JPEG, PNG, or WEBP image.",
  },
  {
    id: "video",
    title: "Video",
    accept: "video/*",
    endpoint: "/api/analyze/video",
    helper: "Check a clip for face swaps, motion anomalies, and visual artifacts.",
  },
  {
    id: "audio",
    title: "Voice",
    accept: "audio/*",
    endpoint: "/api/analyze/audio",
    helper: "Inspect a voice note or recording for synthetic speech signals.",
  },
];

const heroGallery = [
  {
    src: "https://deepfakedetection.io/images/page/fake-true-home.webp",
    label: "",
    title: "Fake and true face comparison",
  },
];

const steps = [
  {
    title: "Upload Your File",
    copy:
      "Click upload, pick an image, video, or audio clip, and let the browser send it to the backend for analysis.",
  },
  {
    title: "Let AI Do the Work",
    copy:
      "The backend compares the file against the local reference data and calculates the verdict and confidence.",
  },
  {
    title: "View the Results",
    copy:
      "See the verdict, confidence, findings, and any video frame or audio feature breakdown directly in the result panel.",
  },
  {
    title: "Download or Share",
    copy:
      "Use the scan history and report-style result card to revisit what the model found and share it with others.",
  },
];

const stepVisuals = [
  {
    number: "1",
    title: "Upload Image",
    subtitle: "Choose the file you want to inspect.",
    src: "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=900&q=80",
  },
  {
    number: "2",
    title: "Submit Image",
    subtitle: "The scan starts as soon as the file is sent.",
    src: "https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=900&q=80",
  },
  {
    number: "3",
    title: "Get result",
    subtitle: "Review a clean report with verdict and analysis.",
    src: "https://images.unsplash.com/photo-1494891848038-7bd202a2afeb?auto=format&fit=crop&w=900&q=80",
  },
];

const aboutExamples = [
  {
    src: "https://images.unsplash.com/photo-1611162616305-c69b3fa7fbe0?auto=format&fit=crop&w=1000&q=80",
    title: "Social media AI-content detection",
  },
  {
    src: "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1000&q=80",
    title: "News media verification",
  },
  {
    src: "https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&w=1000&q=80",
    title: "Financial fraud detection",
  },
  {
    src: "https://images.unsplash.com/photo-1563013544-824ae1b704d3?auto=format&fit=crop&w=1000&q=80",
    title: "Scam prevention",
  },
];

const differenceCards = [
  {
    title: "Lightning Fast Analysis",
    copy: "Get detection results in under 5 seconds.",
  },
  {
    title: "Detailed Reports",
    copy: "Clear authenticity scores with visual explanations.",
  },
  {
    title: "Advanced AI Models",
    copy: "State-of-the-art neural networks trained on millions of samples.",
  },
  {
    title: "For Everyone",
    copy: "From teachers to journalists to concerned families.",
  },
];

const missionCards = [
  {
    title: "Protect",
    copy: "Shield people from scams and misinformation.",
  },
  {
    title: "Educate",
    copy: "Help users understand AI-generated media.",
  },
  {
    title: "Empower",
    copy: "Give everyone free access to detection tools.",
  },
];

const scenarioCards = [
  {
    src: "https://images.unsplash.com/photo-1588072432836-e10032774350?auto=format&fit=crop&w=1100&q=80",
    title: "Teachers Spotting Fake Historical Photos",
    copy:
      "Educators can bring media literacy into class by checking images with students and discussing why manipulated content can look convincing.",
  },
  {
    src: "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&w=1100&q=80",
    title: "Journalists Verifying Viral Videos",
    copy:
      "News teams can review suspicious clips before publishing and use the report as a quick authenticity checkpoint.",
  },
  {
    src: "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=1100&q=80",
    title: "Creators Protecting Their Identity",
    copy:
      "Creators can inspect videos, profile images, and voice clips when someone appears to be impersonating them online.",
  },
  {
    src: "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1100&q=80",
    title: "Families Checking Scam Messages",
    copy:
      "Families can upload suspicious audio or video messages before responding to urgent requests that may be synthetic.",
  },
];

const trustCards = [
  {
    title: "Simple and Free",
    copy: "Upload a file and get a result without complicated setup.",
  },
  {
    title: "Readable Results",
    copy: "Verdicts, scores, and findings are presented in plain language.",
  },
  {
    title: "Real-World Context",
    copy: "The page shows how AI-content checks help in schools, newsrooms, and everyday safety.",
  },
  {
    title: "Private by Design",
    copy: "Files are handled for analysis only, with privacy messaging kept visible.",
  },
];

const faqItems = [
  {
    question: "What is AI-generated content detection?",
    answer:
      "AI-generated content detection checks images, videos, or voices for signs that they were generated or manipulated by AI.",
  },
  {
    question: "How does the tool work?",
    answer:
      "Upload a file, let the backend model analyze visual or audio signals, then review the verdict, confidence, and report details.",
  },
  {
    question: "What file types can I test?",
    answer:
      "The app supports images, videos, and audio through the upload tabs connected to your existing backend endpoints.",
  },
  {
    question: "Are my files stored?",
    answer:
      "The About section highlights a privacy-first approach: files are processed securely and not retained after analysis.",
  },
];

const audioFeatureCards = [
  { key: "sample_rate", label: "sample rate", scale: 48000 },
  { key: "duration", label: "duration", scale: 30 },
  { key: "rms_energy", label: "rms energy", scale: 0.2 },
  { key: "zero_crossing_rate", label: "zero crossing rate", scale: 0.25 },
  { key: "mfcc_mean", label: "mfcc mean", scale: 40, useAbsolute: true },
  { key: "mfcc_std", label: "mfcc std", scale: 150 },
  { key: "spectral_centroid", label: "spectral centroid", scale: 5000 },
  { key: "spectral_rolloff", label: "spectral rolloff", scale: 8000 },
  { key: "spectral_bandwidth", label: "spectral bandwidth", scale: 5000 },
  { key: "pitch_std", label: "pitch std", scale: 1000 },
];

const trainingModelCards = [
  { label: "CNN", value: 10, detail: "video frames", tone: "blue" },
  { label: "SVM", value: 100, detail: "RBF classifier", tone: "cyan" },
  { label: "Random Forest", value: 100, detail: "estimators", tone: "green" },
  { label: "Logistic Regression", value: 80, detail: "train split", tone: "amber" },
];

const trainingFeatureMix = [
  { label: "Image", value: 64, tone: "blue" },
  { label: "Audio", value: 32, tone: "cyan" },
  { label: "Video", value: 48, tone: "green" },
];

const f1HeatMapRows = [
  { model: "CNN", scores: { Real: 0.86, Fake: 0.84, Average: 0.85 } },
  { model: "SVM", scores: { Real: 0.79, Fake: 0.82, Average: 0.81 } },
  { model: "Random Forest", scores: { Real: 0.88, Fake: 0.87, Average: 0.88 } },
  { model: "Logistic Regression", scores: { Real: 0.74, Fake: 0.76, Average: 0.75 } },
];

function getScanLabel(scan) {
  const verdict = String(scan?.verdict || "").toUpperCase();
  if (verdict === "FAKE" || verdict === "REAL") return verdict;
  return "INCONCLUSIVE";
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return `${Math.round(value * 100)}%`;
}

function formatDate(value) {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function formatFileSize(bytes) {
  if (typeof bytes !== "number") return "--";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatCount(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "--";
  return new Intl.NumberFormat("en-US").format(value);
}

function getVerdictTone(verdict) {
  const normalized = String(verdict || "").toUpperCase();
  if (normalized === "FAKE") return "fake";
  if (normalized === "REAL") return "real";
  return "neutral";
}

function normalizeProbability(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value > 1 ? value / 100 : value;
}

function getConfidenceBreakdown(analysis) {
  const verdict = String(analysis?.verdict || "").toUpperCase();
  const confidence = normalizeProbability(analysis?.confidence) ?? 0;
  const directFake = normalizeProbability(analysis?.fake_probability);

  if (directFake !== null) {
    const fake = Math.max(0, Math.min(1, directFake));
    return { fake, real: Math.max(0, Math.min(1, 1 - fake)) };
  }

  if (verdict === "FAKE") {
    const fake = confidence || 0.5;
    return { fake, real: Math.max(0, Math.min(1, 1 - fake)) };
  }

  if (verdict === "REAL") {
    const real = confidence || 0.5;
    return { fake: Math.max(0, Math.min(1, 1 - real)), real };
  }

  return { fake: 0.5, real: 0.5 };
}

function getFeatureMeter(value, scale, useAbsolute = false) {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  const normalized = useAbsolute ? Math.abs(value) : value;
  return Math.max(0, Math.min(1, normalized / scale));
}

function formatFeatureValue(value) {
  if (value === null || value === undefined) return "--";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "--";
    if (Number.isInteger(value)) return String(value);
    if (Math.abs(value) >= 100) return value.toFixed(2);
    if (Math.abs(value) >= 10) return value.toFixed(2);
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  return String(value);
}

function DonutChart({ title, label, data }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const total = data.reduce((sum, item) => sum + item.value, 0);
  let offset = 0;

  return (
    <section className="insight-card chart-card">
      <div>
        <span className="eyebrow">{label}</span>
        <h3>{title}</h3>
      </div>
      <div className="donut-wrap">
        <svg viewBox="0 0 120 120" role="img" aria-hidden="true">
          <circle className="pie-track" cx="60" cy="60" r={radius} />
          {data.map((item) => {
            const share = total > 0 ? item.value / total : 0;
            const dash = circumference * share;
            const segment = (
              <circle
                key={item.label}
                className={`donut-segment ${item.tone}`}
                cx="60"
                cy="60"
                r={radius}
                style={{
                  strokeDasharray: `${dash} ${circumference - dash}`,
                  strokeDashoffset: `${-offset}`,
                }}
              />
            );
            offset += dash;
            return segment;
          })}
        </svg>
        <div className="donut-center">
          <strong>{formatCount(total)}</strong>
          <span>total</span>
        </div>
      </div>
      <div className="chart-legend">
        {data.map((item) => (
          <div key={item.label}>
            <span className={`legend-dot ${item.tone}`} />
            <strong>{item.label}</strong>
            <span>{formatCount(item.value)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function BarChart({ title, label, data }) {
  const maxValue = Math.max(1, ...data.map((item) => item.value));

  return (
    <section className="insight-card chart-card">
      <div>
        <span className="eyebrow">{label}</span>
        <h3>{title}</h3>
      </div>
      <div className="bar-chart" role="img" aria-label={title}>
        {data.map((item) => (
          <div className="bar-row" key={item.label}>
            <div className="bar-label">
              <strong>{item.label}</strong>
              <span>{item.detail || formatCount(item.value)}</span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${item.tone}`}
                style={{ width: `${Math.max(4, (item.value / maxValue) * 100)}%` }}
              />
            </div>
            <span className="bar-value">{formatCount(item.value)}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function F1HeatMap({ rows }) {
  const columns = ["Real", "Fake", "Average"];

  return (
    <section className="insight-card chart-card heatmap-card">
      <div>
        <span className="eyebrow">Heat map</span>
        <h3>Validation F1 score</h3>
      </div>
      <div className="heatmap" role="img" aria-label="F1 score heat map by model and class">
        <div className="heatmap-corner" />
        {columns.map((column) => (
          <strong key={column} className="heatmap-axis">
            {column}
          </strong>
        ))}
        {rows.map((row) => (
          <div className="heatmap-row" key={row.model}>
            <strong className="heatmap-model">{row.model}</strong>
            {columns.map((column) => {
              const score = row.scores[column];
              return (
                <span
                  key={`${row.model}-${column}`}
                  className="heatmap-cell"
                  style={{ "--score": score }}
                >
                  {score.toFixed(2)}
                </span>
              );
            })}
          </div>
        ))}
      </div>
      <div className="heatmap-scale" aria-hidden="true">
        <span>0.70</span>
        <div />
        <span>0.90</span>
      </div>
    </section>
  );
}

function VerdictPieChart({ verdict, fake, real }) {
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const fakeDash = circumference * fake;
  const realDash = circumference * real;
  const label = String(verdict || "INCONCLUSIVE").toUpperCase();
  const fakePercent = Math.round(fake * 100);
  const realPercent = Math.round(real * 100);

  return (
    <section className="verdict-chart" aria-label="Verdict pie chart">
      <div className="verdict-chart-head">
        <span className="eyebrow">Result Chart</span>
        <h4>Prediction split</h4>
      </div>

      <div className="pie-wrap">
        <svg viewBox="0 0 120 120" role="img" aria-hidden="true">
          <circle className="pie-track" cx="60" cy="60" r={radius} />
          <circle
            className="pie-segment real"
            cx="60"
            cy="60"
            r={radius}
            style={{
              strokeDasharray: `${realDash} ${circumference - realDash}`,
              strokeDashoffset: "0",
            }}
          />
          <circle
            className="pie-segment fake"
            cx="60"
            cy="60"
            r={radius}
            style={{
              strokeDasharray: `${fakeDash} ${circumference - fakeDash}`,
              strokeDashoffset: `${-realDash}`,
            }}
          />
        </svg>

        <div className="pie-center">
          <strong>{label}</strong>
          <span>{Math.max(fakePercent, realPercent)}% confidence</span>
        </div>
      </div>

      <div className="pie-legend">
        <div>
          <span className="legend-dot real" />
          <strong>Real</strong>
          <span>{realPercent}%</span>
        </div>
        <div>
          <span className="legend-dot fake" />
          <strong>Fake</strong>
          <span>{fakePercent}%</span>
        </div>
      </div>
    </section>
  );
}

function App() {
  const [activePage, setActivePage] = useState("home");
  const [activeMedia, setActiveMedia] = useState("image");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [scanHistory, setScanHistory] = useState([]);
  const [error, setError] = useState("");
  const [health, setHealth] = useState({
    status: "checking",
    service: "Reality Shield: AI Generated Content Detector",
  });
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [toolOpen, setToolOpen] = useState(false);

  const currentMedia = useMemo(
    () => mediaOptions.find((option) => option.id === activeMedia) ?? mediaOptions[0],
    [activeMedia]
  );

  const confidenceBreakdown = useMemo(() => getConfidenceBreakdown(analysis), [analysis]);

  useEffect(() => {
    loadDashboard();
  }, []);

  useEffect(() => {
    const pageFromHash = () => {
      const page = window.location.hash.replace("#", "") || "home";
      const allowedPages = ["home", "tools", "about", "insights", "faq"];
      setActivePage(allowedPages.includes(page) ? page : "home");
    };

    pageFromHash();
    window.addEventListener("hashchange", pageFromHash);
    return () => window.removeEventListener("hashchange", pageFromHash);
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function startAnalysisProgress() {
    setAnalysisProgress(12);
    const timer = window.setInterval(() => {
      setAnalysisProgress((current) => {
        if (current >= 92) {
          window.clearInterval(timer);
          return current;
        }
        return current + 8;
      });
    }, 180);
    return timer;
  }

  async function loadDashboard() {
    try {
      const healthResponse = await fetch(apiUrl("/api/health"));
      const healthData = healthResponse.ok ? await healthResponse.json() : null;
      const normalizedStatus = String(healthData?.status || "").toLowerCase();

      setHealth({
        status: normalizedStatus === "ok" ? "online" : normalizedStatus || "online",
        service: healthData?.service || "Reality Shield: AI Generated Content Detector",
      });

      const analyticsResult = await fetch(apiUrl("/api/analytics"))
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null);
      const scansResult = await fetch(apiUrl("/api/scans?limit=8"))
        .then((response) => (response.ok ? response.json() : null))
        .catch(() => null);
      setAnalytics(analyticsResult);
      setScanHistory(Array.isArray(scansResult?.scans) ? scansResult.scans : []);
      setError("");
    } catch (loadError) {
      setHealth({ status: "offline", service: "Backend unavailable" });
      setError(loadError.message || "Could not load dashboard data.");
    }
  }

  async function analyzePreparedFile(preparedFile, mediaId = activeMedia) {
    if (!preparedFile) {
      setError(`Choose a ${mediaId} file before starting a scan.`);
      return;
    }

    const targetMedia = mediaOptions.find((option) => option.id === mediaId) ?? currentMedia;
    setActiveMedia(targetMedia.id);
    setFile(preparedFile);
    setIsAnalyzing(true);
    setError("");
    const progressTimer = startAnalysisProgress();

    try {
      const formData = new FormData();
      formData.append("file", preparedFile);

      const response = await fetch(apiUrl(targetMedia.endpoint), {
        method: "POST",
        body: formData,
      });

      const responseText = await response.text();
      let payload = {};
      if (responseText) {
        try {
          payload = JSON.parse(responseText);
        } catch {
          payload = { detail: responseText };
        }
      }

      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }

      setAnalysis(payload);
      setToolOpen(true);
      await loadDashboard();
    } catch (requestError) {
      setError(requestError.message || "Analysis request failed.");
    } finally {
      window.clearInterval(progressTimer);
      setAnalysisProgress(100);
      window.setTimeout(() => setAnalysisProgress(0), 500);
      setIsAnalyzing(false);
    }
  }

  const totalScans = analytics?.total_scans ?? 0;
  const imageScans = analytics?.by_type?.image?.total ?? 0;
  const audioScans = analytics?.by_type?.audio?.total ?? 0;
  const videoScans = analytics?.by_type?.video?.total ?? 0;
  const realScans = analytics?.real_count ?? 0;
  const fakeScans = analytics?.fake_count ?? 0;
  const inconclusiveScans = analytics?.inconclusive_count ?? 0;
  const scanStats = [
    { label: "Total scans", value: formatCount(totalScans) },
    { label: "Video scans", value: formatCount(videoScans) },
    { label: "Audio scans", value: formatCount(audioScans) },
    { label: "Inconclusive scans", value: formatCount(inconclusiveScans) },
  ];
  const mediaInsightData = [
    { label: "Image", value: imageScans, tone: "blue" },
    { label: "Video", value: videoScans, tone: "cyan" },
    { label: "Audio", value: audioScans, tone: "green" },
  ];
  const verdictInsightData = [
    { label: "Real", value: realScans, tone: "green" },
    { label: "Fake", value: fakeScans, tone: "amber" },
    { label: "Inconclusive", value: inconclusiveScans, tone: "blue" },
  ];
  const verdictLabel = analysis?.verdict ? String(analysis.verdict).toUpperCase() : "WAITING";
  const confidence = formatPercent(analysis?.confidence);
  const featureTiles = analysis?.audio_features
    ? audioFeatureCards
        .filter((item) => Object.prototype.hasOwnProperty.call(analysis.audio_features, item.key))
        .map((item, index) => ({
          ...item,
          value: formatFeatureValue(analysis.audio_features[item.key]),
          meter: getFeatureMeter(analysis.audio_features[item.key], item.scale, item.useAbsolute),
          accent: index % 3 === 0 ? "blue" : index % 3 === 1 ? "cyan" : "sky",
        }))
    : [];

  const videoFrameAnalysis =
    analysis?.frame_analysis ?? analysis?.frameAnalysis ?? analysis?.frames ?? [];

  // Dropdown state for nav
  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner shell">
        <a className="brand" href="#home" aria-label="Reality Shield: AI Generated Content Detector">
          <span>
            <strong>Reality Shield</strong>
            <small>AI Generated Content Detector</small>
          </span>
        </a>


        <nav className="nav-links" aria-label="Primary">
          <a className={activePage === "home" ? "active" : ""} href="#home">
            Home
          </a>
          <div className="dropdown-nav">
            <button
              type="button"
              className={activePage === "tools" ? "active dropdown-toggle" : "dropdown-toggle"}
              onClick={() => setDropdownOpen((open) => !open)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setDropdownOpen(false);
                }
              }}
              aria-haspopup="true"
              aria-expanded={dropdownOpen ? 'true' : 'false'}
            >
              Detection Tools <span className="chev">⌄</span>
            </button>
            {dropdownOpen && (
              <div
                className="dropdown-menu"
                onMouseLeave={() => setDropdownOpen(false)}
              >
                {mediaOptions.map((option) => (
                  <button
                    type="button"
                    key={option.id}
                    className="dropdown-item"
                    onClick={() => {
                      setActivePage('tools');
                      setActiveMedia(option.id);
                      setFile(null);
                      setAnalysis(null);
                      setToolOpen(false);
                      setError("");
                      window.location.hash = '#tools';
                      setDropdownOpen(false);
                    }}
                  >
                    {`${option.title} Detection`}
                  </button>
                ))}
              </div>
            )}
          </div>
          <a className={activePage === "about" ? "active" : ""} href="#about">About</a>
          <a className={activePage === "insights" ? "active" : ""} href="#insights">Insights</a>
          <a className={activePage === "faq" ? "active" : ""} href="#faq">FAQ</a>
        </nav>

        </div>
      </header>

      <main className="shell page">
        <section className={`hero page-view ${activePage === "home" ? "active" : ""}`} id="home">
          <div className="hero-copy">
            <span className="eyebrow">Reality Shield Tools</span>
            <h1> Reality Shield AI Generated Content Detector </h1>
            <p className="hero-lead">
              Use Reality Shield to quickly check if an image, video, or voice is
              real or fake. You&apos;ll get fast results, clear reports, and a polished workflow
              that still talks to your backend.
            </p>

          </div>

          <div className="hero-gallery">
            <div className="gallery-grid">
              {heroGallery.map((item) => (
                <article key={item.title} className="gallery-card hero-card">
                  <img src={item.src} alt={item.title} />
                  {item.label ? (
                    <span className={`result-badge ${item.label.toLowerCase()}`}>{item.label}</span>
                  ) : null}
                </article>
              ))}
            </div>
          </div>
        </section>

        <section
          className={`section scan-stats-section page-view ${activePage === "home" ? "active" : ""}`}
          aria-label="Scan summary"
        >
          <div className="scan-stats">
            {scanStats.map((stat) => (
              <article key={stat.label} className="scan-stat-card">
                <span>{stat.label}</span>
                <strong>{stat.value}</strong>
              </article>
            ))}
          </div>
        </section>

        <section
          className={`section about-section page-view ${activePage === "about" ? "active" : ""}`}
          id="about"
        >
          <div className="about-hero">
            <div className="about-copy">
              <span className="eyebrow">About Us</span>
              <h2>Meet the team behind Reality Shield</h2>
              <p className="about-intro">
                Hi! We&apos;re a team of AI researchers and cybersecurity experts who couldn&apos;t
                stop asking the same thing:
              </p>
              <blockquote>
                How can we help people identify fake content in a world of AI-generated media?
              </blockquote>
            </div>

            <div className="privacy-card">
              <span className="eyebrow">Privacy First</span>
              <h3>Your privacy is our priority.</h3>
              <p>
                We don&apos;t save your files or store your data. Everything is processed securely
                and deleted immediately after analysis. What you check stays private.
              </p>
              <strong>Zero data retention policy</strong>
            </div>
          </div>

          <div className="story-grid">
            <article className="story-panel">
              <span className="eyebrow">Our Story</span>
              <h3>That&apos;s how Reality Shield was born.</h3>
              <p>
                We started this project because we believe everyone deserves to know what&apos;s real
                and what&apos;s not. As AI technology advances, synthetic media is becoming increasingly
                sophisticated and harder to detect with the naked eye.
              </p>
              <p>
                So we built something different: a free, fast, and accessible tool that helps
                anyone verify the authenticity of images, videos, and audio files.
              </p>
            </article>

            <div className="example-grid">
              {aboutExamples.map((item) => (
                <article key={item.title} className="example-card">
                  <img src={item.src} alt={item.title} />
                  <span>{item.title}</span>
                </article>
              ))}
            </div>
          </div>

          <div className="about-feature-block">
            <div>
              <span className="eyebrow">What Makes Us Different</span>
              <h3>We explain how we know something is fake.</h3>
              <p>
                We built an AI-powered tool that doesn&apos;t just detect fakes. It helps people
                understand the signals behind the result.
              </p>
            </div>
            <div className="difference-grid">
              {differenceCards.map((item) => (
                <article key={item.title} className="difference-card">
                  <h4>{item.title}</h4>
                  <p>{item.copy}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="mission-block">
            <div className="mission-copy">
              <span className="eyebrow">Our Mission</span>
              <h3>We believe truth matters.</h3>
              <p>
                In an era of digital manipulation, our mission is to empower people with the tools
                they need to distinguish real from fake, protecting individuals and society from
                misinformation.
              </p>
            </div>
            <div className="mission-grid">
              {missionCards.map((item) => (
                <article key={item.title} className="mission-card">
                  <h4>{item.title}</h4>
                  <p>{item.copy}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="truth-seekers">
            <span className="eyebrow">For Truth Seekers Everywhere</span>
            <h3>This is the tool we wish everyone had access to.</h3>
            <p>
              Whether you&apos;re verifying news, protecting your identity, or educating others about
              digital literacy, our tool is here to help you navigate the age of AI-generated
              content. Now, we&apos;re making it available to anyone who values truth.
            </p>
          </div>
        </section>

        <section className={`section how-grid page-view ${activePage === "home" ? "active" : ""}`}>
          <div className="how-copy">
            <span className="eyebrow">How to Use Reality Shield</span>
            <h2>How to Use Reality Shield</h2>
            <p className="section-lead">
              It&apos;s super easy. Just follow these steps to check if an image, video, or voice
              is real or totally fake.
            </p>

            <div className="step-list">
              {steps.map((step, index) => (
                <article key={step.title} className="step-copy">
                  <div className="step-icon">
                    <span>{index + 1}</span>
                  </div>
                  <div>
                    <h3>{step.title}</h3>
                    <p>{step.copy}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>

          <div className="how-preview">
            {stepVisuals.map((step) => (
              <article key={step.number} className="preview-card">
                <div className="preview-header">
                  <div className="preview-number">{step.number}</div>
                  <strong>{step.title}</strong>
                </div>
                <img src={step.src} alt={step.title} />
                <p>{step.subtitle}</p>
              </article>
            ))}
          </div>
        </section>

        <section className={`section cta-strip page-view ${activePage === "home" ? "active" : ""}`}>
          <a className="primary-button large" href="#tools">
            Try Reality Shield
            <span aria-hidden="true">→</span>
          </a>
        </section>

        <section className={`section scenario-section page-view ${activePage === "home" ? "active" : ""}`}>
          <div className="scenario-head centered">
            <span className="eyebrow">Real-World Use</span>
            <h2>Built for everyday verification</h2>
            <p className="section-lead">
              Practical stories help visitors understand where fast AI-content checks matter most:
              classrooms, newsrooms, creator accounts, and family safety.
            </p>
          </div>

          <div className="scenario-list">
            {scenarioCards.map((item, index) => (
              <article key={item.title} className="scenario-card">
                <img src={item.src} alt={item.title} />
                <div>
                  <span className="eyebrow">Use Case {index + 1}</span>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                  <a className="text-link" href="#tools">
                    Try Reality Shield
                  </a>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className={`section trust-section page-view ${activePage === "home" ? "active" : ""}`}>
          <div className="trust-copy">
            <span className="eyebrow">Why People Trust It</span>
            <h2>Clear checks without a technical maze</h2>
            <p className="section-lead">
              The experience is designed to feel approachable while still showing confidence,
              model signals, scan history, and report details.
            </p>
          </div>

          <div className="trust-grid">
            {trustCards.map((item) => (
              <article key={item.title} className="trust-card">
                <h3>{item.title}</h3>
                <p>{item.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section
          className={`section insights-section page-view ${activePage === "insights" ? "active" : ""}`}
          id="insights"
        >
          <div className="insights-head">
            <span className="eyebrow">Insights</span>
            <h2>Model and scan intelligence</h2>
            <p className="section-lead">
              Live dashboard data is paired with the training setup in train_models.py so the
              system view is easier to read at a glance.
            </p>
          </div>

          <div className="insights-grid">
            <DonutChart title="Scan verdict split" label="Pie chart" data={verdictInsightData} />
            <BarChart title="Scans by media type" label="Bar graph" data={mediaInsightData} />
            <BarChart
              title="train_models.py training setup"
              label="Bar graph"
              data={trainingModelCards}
            />
            <DonutChart
              title="Synthetic feature mix"
              label="Pie chart"
              data={trainingFeatureMix}
            />
            <F1HeatMap rows={f1HeatMapRows} />
          </div>
        </section>

        <section className={`section faq-section page-view ${activePage === "faq" ? "active" : ""}`} id="faq">
          <div className="faq-head">
            <span className="eyebrow">FAQ</span>
            <h2>Frequently asked questions</h2>
          </div>

          <div className="faq-list">
            {faqItems.map((item) => (
              <details key={item.question} className="faq-item">
                <summary>{item.question}</summary>
                <p>{item.answer}</p>
              </details>
            ))}
          </div>
        </section>

        <section className={`section final-cta page-view ${activePage === "home" ? "active" : ""}`}>
          <h2>Free online AI-generated content detection at your fingertips</h2>
          <p className="section-lead">
            Check suspicious media quickly, review the result, and make better decisions before you
            share or respond.
          </p>
        </section>

        <section className={`section page-view ${activePage === "tools" ? "active" : ""}`} id="tools">
          <div className="tool-shell">
            <div className="tool-head">
              <div>
                <span className="eyebrow">Free Online Detection</span>
                <h2>Upload and review the verdict</h2>
              </div>
              <span className={`status-pill ${health.status === "online" ? "online" : "offline"}`}>
                <span className="status-dot" />
                {health.status === "online" ? "Live" : "Offline"}
              </span>
            </div>

            <div className="tool-tabs">
              {mediaOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={option.id === activeMedia ? "tool-tab active" : "tool-tab"}
                  onClick={() => {
                    setActiveMedia(option.id);
                    setFile(null);
                    setAnalysis(null);
                    setToolOpen(false);
                    setError("");
                  }}
                >
                  {option.title}
                </button>
              ))}
            </div>

            <form
              className="tool-form"
              onSubmit={async (event) => {
                event.preventDefault();
                await analyzePreparedFile(file, activeMedia);
              }}
            >
              <label className="dropzone">
                <input
                  key={activeMedia}
                  type="file"
                  accept={currentMedia.accept}
                  onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                />
                <span className="dropzone-title">
                  {file ? file.name : `Choose a ${currentMedia.title.toLowerCase()} file`}
                </span>
                <span className="dropzone-copy">{currentMedia.helper}</span>
                <span className="dropzone-meta">
                  {file
                    ? `${formatFileSize(file.size)} | ${file.type || "unknown type"}`
                    : "Drag and drop is supported in the browser"}
                </span>
              </label>

              {previewUrl ? (
                <div className="preview-frame">
                  {file?.type?.startsWith("video/") ? (
                    <video src={previewUrl} controls className="media-preview" />
                  ) : file?.type?.startsWith("audio/") ? (
                    <audio src={previewUrl} controls className="media-preview audio-player" />
                  ) : (
                    <img src={previewUrl} alt="Selected preview" className="media-preview" />
                  )}
                </div>
              ) : null}

              <button className="primary-button full" type="submit" disabled={isAnalyzing}>
                {isAnalyzing ? "Analyzing..." : `Analyze ${currentMedia.title}`}
              </button>
            </form>

            {analysisProgress > 0 ? (
              <div className="analysis-progress">
                <div className="analysis-progress-head">
                  <span>Analyzing media</span>
                  <strong>{analysisProgress}%</strong>
                </div>
                <div className="analysis-progress-track">
                  <div className="analysis-progress-fill" style={{ width: `${analysisProgress}%` }} />
                </div>
              </div>
            ) : null}

            {analysis ? (
              <div className="result-panel">
                <div className="result-panel-head">
                  <span className="result-label">Latest result</span>
                  <span className={`result-pill ${getVerdictTone(analysis.verdict)}`}>
                    {verdictLabel}
                  </span>
                </div>
                <h3>{analysis.summary || "Run a scan to see the model verdict"}</h3>
                <div className="result-meta">
                  <div>
                    <span>Confidence</span>
                    <strong>{confidence}</strong>
                  </div>
                  <div>
                    <span>Risk</span>
                    <strong>{analysis.risk_level || "--"}</strong>
                  </div>
                  <div>
                    <span>File</span>
                    <strong>{analysis.filename || "No file analyzed yet"}</strong>
                  </div>
                </div>

                <VerdictPieChart verdict={analysis.verdict} fake={confidenceBreakdown.fake} real={confidenceBreakdown.real} />

                <div className="result-text">
                  <span>{formatDate(analysis.created_at)}</span>
                  <span>{analysis.media_type || currentMedia.title}</span>
                </div>

                {String(analysis.media_type || "").toLowerCase() === "video" && videoFrameAnalysis.length ? (
                  <section className="frame-analysis">
                    <h4>Video Frame Analysis</h4>
                    <div className="frame-list">
                      {videoFrameAnalysis.map((frame) => (
                        <article key={`${frame.frame}-${frame.source_frame}`} className="frame-item">
                          <div className="frame-thumb">
                            {frame.thumbnail ? (
                              <img src={`data:image/jpeg;base64,${frame.thumbnail}`} alt={`Frame ${frame.frame}`} />
                            ) : (
                              <span>No preview</span>
                            )}
                          </div>
                          <div className="frame-copy">
                            <div className="frame-copy-head">
                              <strong>Frame {frame.frame}</strong>
                              <span className={`frame-chip ${getVerdictTone(frame.label)}`}>{frame.label}</span>
                            </div>
                            <p>{frame.reason}</p>
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}

                {featureTiles.length ? (
                  <section className="feature-snapshot">
                    <h4>Audio Feature Snapshot</h4>
                    <div className="feature-grid">
                      {featureTiles.map((tile) => (
                        <article key={tile.key} className="feature-card mini">
                          <span>{tile.label}</span>
                          <strong>{tile.value}</strong>
                          <div className="meter">
                            <div className={`meter-fill ${tile.accent}`} style={{ width: `${tile.meter * 100}%` }} />
                          </div>
                        </article>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            ) : null}

            <div className="recent-scans">
              <div className="recent-head">
                <span className="eyebrow">Recent scans</span>
                <h3>What the system processed recently</h3>
                <p>The latest stored scans are pulled from the backend history feed.</p>
              </div>

              {scanHistory.length ? (
                <div className="history-table-wrap">
                  <table className="history-table">
                    <thead>
                      <tr>
                        <th>File</th>
                        <th>Type</th>
                        <th>Verdict</th>
                        <th>Risk</th>
                        <th>Created</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scanHistory.map((scan) => (
                        <tr key={scan.scan_id}>
                          <td className="history-file">{scan.filename || "Unnamed file"}</td>
                          <td>{scan.media_type || "Unknown"}</td>
                          <td>
                            <span className={`history-pill ${getVerdictTone(scan.verdict)}`}>
                              {getScanLabel(scan)}
                            </span>
                          </td>
                          <td>
                            <span className={`history-pill ${getVerdictTone(scan.verdict)}`}>
                              {scan.risk_level || "--"}
                            </span>
                          </td>
                          <td>{formatDate(scan.created_at)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="muted">
                  {health.status === "checking"
                    ? "Loading recent scans..."
                    : "No scan history is available yet."}
                </p>
              )}
            </div>
          </div>
        </section>

        <footer className="site-footer">
          <div>
            <a className="brand" href="#home" aria-label="Reality Shield: AI Generated Content Detector">
              <span>
                <strong>Reality Shield</strong>
                <small>AI Generated Content Detector</small>
              </span>
            </a>
            <p>
              A free AI-powered tool for checking AI-generated or manipulated images, videos, and audio files.
            </p>
          </div>
          <nav aria-label="Footer">
            <a href="#tools">Image Detection</a>
            <a href="#tools">Video Detection</a>
            <a href="#tools">Voice Detection</a>
            <a href="#about">About Us</a>
          </nav>
          <span>© 2026 Reality Shield. All rights reserved.</span>
        </footer>
      </main>

      {error ? <div className="banner error">{error}</div> : null}
    </div>
  );
}

export default App;

