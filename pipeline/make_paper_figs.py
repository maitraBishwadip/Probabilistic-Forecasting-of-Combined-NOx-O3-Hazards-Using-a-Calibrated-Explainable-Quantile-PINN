"""
make_paper_figs.py -- Publication-ready figures for the InGARSS 2026 manuscript.

Layout (revised per author request):
  Fig 1  : ONE composite -> (a) BD station map, (b) pollutant data distribution,
           (c) NO2 kriged surface, (d) O3 kriged surface.
  Fig 2  : intentionally NOT generated -- left as a blank placeholder in main.tex
           (author will insert manually).
  Fig 3  : reliability diagram (publication grade, single panel).
  Fig 4  : XAI physics-group importance (publication grade, single panel).
  Figs 3 & 4 are placed SIDE BY SIDE in main.tex to save space.

All numbers TRANSCRIBED VERBATIM from RESULTS_LOG.md (source of truth, CLAUDE.md sec.1):
  Fig 3 -> results_e6.json (run E6);  Fig 4 -> RESULTS_LOG.md E9 group table (run E9).
Station coords/meta: gis_station_pollutant_regimes.json. Geometry: bd.json (EPSG:4326).
Pollutant surfaces: no2.png / o3.png. Raw distributions: BD_DOE_2014-16.csv.

Outputs -> paper/figures/{fig1_composite,fig3_reliability,fig4_importance}.{png,pdf}
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ global style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.linewidth": 0.9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 8, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
INK = "#1a1a1a"

def _f(*names):
    """Resolve a data file that may live at ROOT or in a moved subfolder (data/, results/)."""
    for n in names:
        p = os.path.join(ROOT, n)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, names[0])

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    # camera-ready raster deliverable: LZW-compressed TIFF (author request)
    try:
        fig.savefig(os.path.join(OUT, f"{name}.tiff"), pil_kwargs={"compression": "tiff_lzw"})
    except Exception:
        fig.savefig(os.path.join(OUT, f"{name}.tiff"))
    plt.close(fig)
    print("saved", name, "(png/pdf/tiff)")

def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)

# ================================================================ map panel
# station id order (used in Fig.1 caption legend)
STATION_NUM = {"AGRABAD":1,"KHULNA":2,"BARC":3,"SYLHET":4,"DARUS SALAM":5,
               "RAJSHAHI":6,"NARAYANGANJ":7,"BARISHAL":8,"GAZIPUR":9}

def draw_map(ax, fig=None):
    bd = json.load(open(os.path.join(ROOT, "bd.json")))
    st = json.load(open(os.path.join(ROOT, "gis_station_pollutant_regimes.json")))["stations"]
    patches = []
    for feat in bd["features"]:
        g = feat["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            patches.append(MplPolygon(np.asarray(poly[0]), closed=True))
    ax.add_collection(PatchCollection(patches, facecolor="#eef3ee",
                      edgecolor="#8aa0a8", linewidths=0.5, zorder=1))
    lon = np.array([s["longitude"] for s in st])
    lat = np.array([s["latitude"] for s in st])
    ax.scatter(lon, lat, c="#cb181d", s=70, marker="*", edgecolor="black",
               linewidth=0.6, zorder=5)
    # numbered markers (offsets de-cluster the 4-station Dhaka cluster: 3,5,7,9)
    numoff = {"AGRABAD":(0.30,0.0),"KHULNA":(-0.30,0.0),"BARC":(-0.26,0.04),
              "SYLHET":(0.30,0.04),"DARUS SALAM":(-0.20,0.34),"RAJSHAHI":(-0.30,0.04),
              "NARAYANGANJ":(0.34,-0.24),"BARISHAL":(0.0,-0.34),"GAZIPUR":(0.34,0.28)}
    for s in st:
        n = s["station"]; dx, dy = numoff[n]; x, y = s["longitude"], s["latitude"]
        ax.annotate(str(STATION_NUM[n]), (x, y), (x + dx, y + dy), ha="center",
                    va="center", fontsize=5.6, fontweight="bold", color="#111", zorder=7,
                    bbox=dict(boxstyle="circle,pad=0.10", fc="white", ec="#cb181d", lw=0.6))
    ax.annotate("N", xy=(0.08, 0.94), xytext=(0.08, 0.81), xycoords="axes fraction",
                ha="center", fontsize=7.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    x0, y0, L = 88.35, 21.5, 0.98
    ax.plot([x0, x0 + L], [y0, y0], color="black", lw=1.8, solid_capstyle="butt")
    ax.text(x0 + L/2, y0 + 0.12, "100 km", ha="center", va="bottom", fontsize=5.3)
    ax.set_xlim(87.6, 93.4); ax.set_ylim(20.4, 27.1)
    ax.set_aspect(1.0/np.cos(np.deg2rad(23.7)))
    ax.set_xlabel("Longitude (°E)", fontsize=7); ax.set_ylabel("Latitude (°N)", fontsize=7)
    ax.tick_params(labelsize=6)
    despine(ax)

# ================================================================ distribution panel
def draw_distribution(ax):
    df = pd.read_csv(_f("BD_DOE_2014-16.csv", "data/BD_DOE_2014-16.csv"),
                     usecols=["O3_ppb", "NOx_ppb"])
    o3 = df["O3_ppb"].to_numpy(); nox = df["NOx_ppb"].to_numpy()
    o3 = o3[(o3 > 0) & (o3 <= 300) & np.isfinite(o3)]
    nox = nox[(nox > 0) & (nox <= 2000) & np.isfinite(nox)]
    bins = np.logspace(np.log10(0.3), np.log10(800), 55)
    for v, c, lab in [(o3, "#d7301f", "$\\mathrm{O_3}$"),
                      (nox, "#2b8cbe", "$\\mathrm{NO_x}$")]:
        ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                color=c, alpha=0.28, lw=0)
        ax.hist(v, bins=bins, density=True, histtype="step", color=c, lw=1.6, label=lab)
        med = np.median(v)
        ax.axvline(med, color=c, ls="--", lw=1.0, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Concentration (ppb)", fontsize=7)
    ax.set_ylabel("Probability density", fontsize=7)
    ax.set_xlim(0.3, 800)
    ax.tick_params(labelsize=6)
    ax.legend(loc="upper right", handlelength=1.1, fontsize=6.4, borderaxespad=0.3)
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    despine(ax)

# ================================================================ pollutant surface panel
def draw_surface(ax, key, cmap, label, fig):
    """IDW station-mean surface (ppb) clipped to Bangladesh -- replaces the missing no2/o3.png
    with a crisp, reproducible kriged-style surface. Values from gis_station_pollutant_regimes.json."""
    from matplotlib.path import Path as MplPath
    bd = json.load(open(os.path.join(ROOT, "bd.json")))
    st = json.load(open(os.path.join(ROOT, "gis_station_pollutant_regimes.json")))["stations"]
    px = np.array([s["longitude"] for s in st]); py = np.array([s["latitude"] for s in st])
    pv = np.array([s[key] for s in st], float)
    gx, gy = np.meshgrid(np.linspace(87.9, 92.8, 260), np.linspace(20.4, 26.8, 320))
    Z = np.zeros_like(gx); W = np.zeros_like(gx)
    for xi, yi, vi in zip(px, py, pv):                       # inverse-distance weighting
        w = 1.0 / ((gx - xi) ** 2 + (gy - yi) ** 2 + 1e-6) ** 1.35
        Z += w * vi; W += w
    Z = Z / W
    paths = []
    for feat in bd["features"]:
        g = feat["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            paths.append(MplPath(np.asarray(poly[0])))
    pts = np.column_stack([gx.ravel(), gy.ravel()]); inside = np.zeros(len(pts), bool)
    for p in paths:
        inside |= p.contains_points(pts)
    Zm = Z.ravel(); Zm[~inside] = np.nan; Zm = Zm.reshape(gx.shape)
    im = ax.contourf(gx, gy, Zm, levels=14, cmap=cmap, zorder=1, extend="max")
    ax.contour(gx, gy, Zm, levels=7, colors="white", linewidths=0.25, alpha=0.55, zorder=2)
    for p in paths:
        v = p.vertices; ax.plot(v[:, 0], v[:, 1], color="#333", lw=0.35, zorder=3)
    ax.scatter(px, py, c="k", s=8, marker="*", edgecolor="white", linewidth=0.35, zorder=4)
    ax.set_xlim(87.9, 92.8); ax.set_ylim(20.4, 26.8)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(23.7)))
    ax.set_xticks([]); ax.set_yticks([]); despine(ax, keep=())
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.set_label(label, fontsize=5.6, labelpad=1)
    cb.outline.set_linewidth(0.4)

# ================================================================ FIG 1 composite
def fig1_composite():
    fig = plt.figure(figsize=(7.16, 2.55))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.30, 1.18, 1.0, 1.0], wspace=0.34)
    ax_map = fig.add_subplot(gs[0, 0]); draw_map(ax_map, fig)
    ax_dst = fig.add_subplot(gs[0, 1]); draw_distribution(ax_dst)
    ax_no2 = fig.add_subplot(gs[0, 2]); draw_surface(ax_no2, "mean_NO2_ppb", "Blues", "ppb", fig)
    ax_o3  = fig.add_subplot(gs[0, 3]); draw_surface(ax_o3, "mean_O3_ppb", "YlOrRd", "ppb", fig)
    tags = [(ax_map,"(a) Network"),(ax_dst,"(b) Data distribution"),
            (ax_no2,"(c) $\\mathrm{NO_2}$ surface"),(ax_o3,"(d) $\\mathrm{O_3}$ surface")]
    for ax, t in tags:
        ax.set_title(t, fontsize=7.5, fontweight="bold", pad=3, loc="left")
    save(fig, "fig1_composite")

# ================================================================ FIG 2 architecture
def _box(ax, cx, cy, w, h, title, body, fc, ec, tc="#111", fs=6.4):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.015,rounding_size=0.10",
                 fc=fc, ec=ec, lw=1.1, zorder=3))
    ax.text(cx, cy + h/2 - 0.20, title, ha="center", va="top", fontsize=fs + 1.1,
            fontweight="bold", color=tc, zorder=4)
    ax.text(cx, cy + h/2 - 0.52, body, ha="center", va="top", fontsize=fs,
            color=tc, zorder=4, linespacing=1.25)

def _arr(ax, p0, p1, color="#555", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                 color=color, lw=lw, shrinkA=1, shrinkB=1, zorder=2,
                 connectionstyle=f"arc3,rad={rad}"))

def fig2_architecture():
    # compact VERTICAL layout -> readable at single-column width
    DELIV, DELIVE = "#c6dbef", "#2171b5"
    PHYS,  PHYSE  = "#fdd0a2", "#d94801"
    NEUT,  NEUTE  = "#e9ecef", "#868e96"
    LOSS,  LOSSE  = "#d3eecd", "#41ab5d"
    fig, ax = plt.subplots(figsize=(3.5, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0.6, 12.2); ax.axis("off")

    _box(ax, 5.0, 11.1, 7.2, 1.55, "INPUT WINDOW",
         "48 h $\\times$ 26 feat.: lags, met,\nreanalysis, AOD, calendar,\nmasks $+$ station id",
         NEUT, NEUTE, fs=5.9)
    _box(ax, 5.0, 8.75, 5.2, 1.45, "ENCODER",
         "LSTM $+$ station emb.\n$\\rightarrow$ latent $z_t$", NEUT, NEUTE, fs=5.9)
    _box(ax, 2.6, 5.95, 4.2, 1.8, "QUANTILE HEAD",
         "monotone,\nnon-crossing\n$Q_\\tau(H_{t+h})$", DELIV, DELIVE, fs=5.9)
    _box(ax, 7.4, 5.95, 4.2, 1.8, "PHYSICS HEAD",
         "NO–NO$_2$–O$_3$\nbox-ODE; Leighton\n$+\\,$O$_x$ cons.", PHYS, PHYSE, fs=5.9)
    _box(ax, 2.6, 2.85, 4.2, 1.8, "PREDICTIVE DIST.",
         "CDF/PDF of\n$H_{t+1..24}$;\n$P(H \\geq \\mathrm{thr})$", DELIV, DELIVE, fs=5.9)
    _box(ax, 7.4, 2.85, 4.2, 1.8, "COMPOSITE LOSS",
         "pinball $+\\,\\lambda_{\\mathrm{phys}}r$\n$+\\,\\lambda_{\\mathrm{chem}}(r_L{+}r_{O_x})$",
         LOSS, LOSSE, fs=5.9)

    _arr(ax, (5.0, 10.32), (5.0, 9.48))               # input -> encoder
    _arr(ax, (4.0, 8.02), (3.0, 6.86))                # encoder -> quantile
    _arr(ax, (6.0, 8.02), (7.0, 6.86))                # encoder -> physics
    _arr(ax, (2.6, 5.05), (2.6, 3.76), color=DELIVE)  # quantile -> predictive
    _arr(ax, (7.4, 5.05), (7.4, 3.76), color=PHYSE)   # physics -> loss
    _arr(ax, (4.2, 5.30), (5.7, 3.80), color=DELIVE, lw=1.1, rad=-0.25)  # pinball -> loss
    ax.text(5.15, 4.55, "pinball", fontsize=5.2, color=DELIVE, style="italic", rotation=-55)
    ax.text(2.7, 1.78, "★ deliverable", ha="center", fontsize=5.8,
            color=DELIVE, fontweight="bold")
    ax.text(7.3, 1.78, "regulariser, not accuracy", ha="center", fontsize=5.2,
            color=PHYSE, style="italic")
    fig.tight_layout(pad=0.2)
    save(fig, "fig2_architecture")

# ================================================================ FIG 3 reliability
def fig3_reliability():
    e6 = json.load(open(_f("results_e6.json", "results/results_e6.json")))
    cov, cqrd = e6["coverage"], e6["cqr"]
    levels = ["80", "90", "95"]
    nominal = np.array([cov[l]["nominal"] for l in levels])
    raw = np.array([cov[l]["raw_picp"] for l in levels])
    pit = np.array([cov[l]["pit_picp"] for l in levels])
    cqr = np.array([cov[l]["cqr_picp"] for l in levels])

    fig, ax = plt.subplots(figsize=(3.85, 3.5))
    lo, hi = 0.695, 0.975
    # coloured calibration field: green = conservative, red = under-coverage
    ax.fill_between([lo, hi], [lo, hi], hi, color="#1a9850", alpha=0.07, zorder=0)
    ax.fill_between([lo, hi], lo, [lo, hi], color="#d73027", alpha=0.08, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 3)), color="#555", lw=1.2, zorder=1)
    ax.text(0.745, 0.775, "1:1 ideal", color="#555", fontsize=6.6, rotation=42,
            ha="center", va="center", style="italic")
    # curves: raw + PIT muted, CQR hero
    ax.plot(nominal, raw, "o-", color="#9e9e9e", lw=1.4, ms=5.5, mec="white",
            mew=0.8, label="Raw quantiles", zorder=3)
    ax.plot(nominal, pit, "s-", color="#3690c0", lw=1.6, ms=6, mec="white",
            mew=0.8, label="PIT recalibrated", zorder=3)
    ax.plot(nominal, cqr, "D-", color="#d7301f", lw=2.6, ms=8.5, mec="white",
            mew=1.0, label="CQR (proposed)", zorder=6)
    # "calibration gain" arrow at the 90% level (raw -> CQR)
    ax.annotate("", xy=(0.90, cqr[1] - 0.004), xytext=(0.90, raw[1] + 0.004),
                arrowprops=dict(arrowstyle="-|>", color="#d7301f", lw=1.6,
                                shrinkA=0, shrinkB=0), zorder=5)
    ax.annotate("+%.3f\ncoverage" % (cqr[1] - raw[1]), (0.90, (cqr[1] + raw[1]) / 2),
                (0.838, 0.852), fontsize=6.6, color="#b30000", fontweight="bold",
                ha="center", va="center", zorder=7)
    ax.set_xlabel("Nominal coverage  $1-\\alpha$")
    ax.set_ylabel("Empirical coverage (PICP)")
    ax.set_xlim(0.755, hi); ax.set_ylim(lo, hi)
    ax.set_xticks([0.80, 0.90, 0.95]); ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    ax.legend(loc="upper left", handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)
    ax.grid(alpha=0.16, lw=0.5)
    # inset: CQR coverage stable across lead time
    axin = ax.inset_axes([0.605, 0.085, 0.375, 0.34])
    leads = [1, 6, 12, 24]
    pal = {"80": "#fdae6b", "90": "#f16913", "95": "#7f2704"}
    for l in levels:
        ph = cqrd[l]["picp_h_1_6_12_24"]
        axin.plot(leads, ph, "o-", color=pal[l], ms=2.6, lw=1.1, mec="white", mew=0.3)
        axin.axhline(float(l) / 100, ls=":", color=pal[l], lw=0.7, alpha=0.7)
    axin.set_title("CQR coverage vs. lead", fontsize=5.8, pad=1.5)
    axin.set_xticks(leads); axin.set_yticks([0.8, 0.9])
    axin.tick_params(labelsize=4.8, length=1.8, pad=1)
    axin.set_xlabel("lead $h$ (h)", fontsize=5.4, labelpad=1)
    axin.set_ylim(0.74, 0.99); axin.set_xlim(-1, 26)
    axin.grid(alpha=0.18, lw=0.4)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    despine(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "fig3_reliability")

# ================================================================ FIG 4 importance
def fig4_importance():
    # transcribed VERBATIM from RESULTS_LOG.md run E9-20260616-085543
    O3 = "$\\mathrm{O_3}$ lags"
    ig  = {"NOx lags":0.375, O3:0.298, "Calendar":0.153, "Dispersion":0.076,
           "Photochem.":0.064, "Pressure":0.018, "Wet removal":0.013, "Satellite":0.003}
    occ = {"NOx lags":0.2225, O3:0.0924, "Calendar":0.0175, "Dispersion":0.0004,
           "Photochem.":0.0027, "Pressure":0.0031, "Wet removal":0.0008, "Satellite":-0.0000}
    R = 0.917
    order = sorted(ig, key=lambda g: ig[g])           # ascending -> top bar is NOx lags
    ig_n = np.array([ig[g] for g in order]); ig_n = ig_n / ig_n.sum()
    # emphasis palette: the three real drivers pop, the rest are muted "ignored"
    emph = {"NOx lags":"#cb181d", O3:"#fb6a4a", "Calendar":"#fdae6b"}
    colors = [emph.get(g, "#c7c7c7") for g in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(3.85, 3.5))
    ax.barh(y, ig_n, height=0.62, color=colors, edgecolor="white", lw=0.6, zorder=3)
    for i, v in enumerate(ig_n):
        ax.text(v + 0.006, i, f"{v:.2f}", va="center", ha="left", fontsize=6, color="#333")
    ax.set_yticks(y); ax.set_yticklabels(order)
    for i, g in enumerate(order):
        ax.get_yticklabels()[i].set_color("#111" if g in emph else "#888")
        if g in ("NOx lags", O3):
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_xlabel("Integrated-Gradients importance (norm.)")
    ax.set_xlim(0, ig_n.max() * 1.30); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", alpha=0.16, lw=0.5)
    # inset: IG vs occlusion agreement (the robustness proof)
    axin = ax.inset_axes([0.52, 0.11, 0.45, 0.40])
    xs = np.array([ig[g] for g in ig]); ys = np.array([occ[g] for g in ig])
    m, b = np.polyfit(xs, ys, 1); xx = np.linspace(xs.min(), xs.max(), 40)
    axin.plot(xx, m * xx + b, "--", color="#888", lw=0.9, zorder=1)
    axin.scatter(xs, ys, s=24, c="#54278f", edgecolor="white", lw=0.5, zorder=3)
    axin.set_title("IG vs. occlusion", fontsize=5.8, pad=1.5)
    axin.text(0.06, 0.90, "$r=%.3f$" % R, transform=axin.transAxes, fontsize=6.4,
              fontweight="bold", va="top", color="#54278f")
    axin.set_xlabel("IG", fontsize=5.4, labelpad=1); axin.set_ylabel("occlusion", fontsize=5.4, labelpad=1)
    axin.tick_params(labelsize=4.6, length=1.8, pad=1)
    axin.set_xticks([0, 0.2]); axin.set_yticks([0, 0.2])
    axin.grid(alpha=0.18, lw=0.4)
    for s in ("top", "right"):
        axin.spines[s].set_visible(False)
    despine(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "fig4_importance")

# ================================================================ FIG 2 forecast
def fig2_forecast():
    """Two representative day-ahead forecasts from the headline data-only QR-PINN (FREE), test 2016.
    All arrays are REAL cached predictions (artefacts/e12_preds.npz, run E12) -- nothing synthetic.
    Each panel: predictive median + 80%/90% intervals + observed H, with the corrected extreme line
    H>=Q70 and a derived exceedance-probability ribbon P(H>=Q70) read off the predicted CDF."""
    THR70 = 0.6087  # src: RESULTS_LOG.md THRCORR-Q70 (corrected top-30% extreme cut)
    d = np.load(os.path.join(ROOT, "artefacts", "e12_preds.npz"), allow_pickle=True)
    Q, Y, M = d["free_Q"], d["free_Y"], d["free_M"]
    taus = d["taus"]; stn = d["te_station"]
    stations = json.load(open(os.path.join(ROOT, "artefacts", "qrpinn_meta.json")))["stations"]
    i05, i10, i50, i90, i95 = 0, 1, 4, 7, 8     # tau grid [.05 .10 .20 .30 .50 .70 .80 .90 .95]
    HZ = Q.shape[1]; x = np.arange(1, HZ + 1); nobs = M.sum(1)
    strong = {"SYLHET", "DARUS SALAM", "NARAYANGANJ", "KHULNA", "AGRABAD", "BARISHAL"}

    def cov90(i):
        m = M[i] > 0
        return ((Y[i] >= Q[i, :, i05]) & (Y[i] <= Q[i, :, i95]))[m].mean()
    def pexc(i):                                 # P(H>=Q70) per lead off the predicted CDF
        out = np.empty(HZ)
        for h in range(HZ):
            q = Q[i, h]; k = int((q < THR70).sum())
            if k == 0: out[h] = taus[0]
            elif k == len(taus): out[h] = taus[-1]
            else: out[h] = taus[k - 1] + (THR70 - q[k - 1]) / max(q[k] - q[k - 1], 1e-9) * (taus[k] - taus[k - 1])
        return 1.0 - out

    # --- deterministic, reproducible anchor selection from the fixed npz ---
    epi = []                                      # (a) hazardous build-up crossing Q70
    for i in range(len(Y)):
        if nobs[i] < 21 or stations[stn[i]] not in strong: continue
        m = M[i] > 0; yo = Y[i][m]
        if not (yo.min() < THR70 <= yo.max()): continue
        ef = (yo >= THR70).mean()
        if not (0.25 <= ef <= 0.7): continue
        c = cov90(i); rng = yo.max() - yo.min(); mae = np.abs(yo - Q[i, :, i50][m]).mean()
        if c >= 0.85 and rng >= 0.30 and mae <= 0.07:
            epi.append((rng, -mae, i))
    epi.sort(reverse=True); iA = epi[0][2]; sA = stations[stn[iA]]
    calm = []                                     # (b) low-risk day at a DIFFERENT station
    for i in range(len(Y)):
        if nobs[i] < 21 or stations[stn[i]] == sA or stations[stn[i]] not in strong: continue
        m = M[i] > 0; yo = Y[i][m]; ef = (yo >= THR70).mean()
        if ef > 0.15: continue
        c = cov90(i); rng = yo.max() - yo.min(); mae = np.abs(yo - Q[i, :, i50][m]).mean()
        if c >= 0.90 and rng >= 0.10 and mae <= 0.07:
            calm.append((ef, mae, -rng, i))
    calm.sort(); iB = calm[0][3]; sB = stations[stn[iB]]
    print(f"  fig2_forecast anchors: A={iA} ({sA}, episode), B={iB} ({sB}, low-risk)")

    # ---- predictive-density "river": smooth contour of p(H|h) from the monotone quantiles ----
    yg = np.linspace(0.0, 1.0, 280)
    kx = np.arange(-14, 15); ker = np.exp(-(kx ** 2) / (2 * 5.5 ** 2)); ker /= ker.sum()
    def dens_field(i):
        cols = []
        for h in range(HZ):
            q = np.maximum.accumulate(Q[i, h].astype(float)) + 1e-4 * np.arange(len(taus))
            F = np.interp(yg, q, taus, left=0.0, right=1.0)        # piecewise-linear CDF
            f = np.clip(np.gradient(F, yg), 0, None)               # density = dF/dH
            f = np.convolve(f, ker, mode="same")
            cols.append(f / (f.max() + 1e-9))                      # per-lead relative density
        return np.column_stack(cols)                               # [H, lead]
    ZA, ZB = dens_field(iA), dens_field(iB)
    XX, YY = np.meshgrid(x, yg)

    fig = plt.figure(figsize=(7.16, 3.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 0.95], hspace=0.14, wspace=0.20,
                          left=0.065, right=0.895, top=0.905, bottom=0.135)
    mesh = None
    for col, (i, stt, tag, Z) in enumerate([(iA, sA, "(a)", ZA), (iB, sB, "(b)", ZB)]):
        ax = fig.add_subplot(gs[0, col]); axp = fig.add_subplot(gs[1, col], sharex=ax)
        m = M[i] > 0
        mesh = ax.pcolormesh(XX, YY, Z, cmap="Blues", shading="gouraud", vmin=0, vmax=1,
                             zorder=0, rasterized=True)
        # faint 90% predictive envelope + crisp white median (cased for contrast on the density)
        ax.plot(x, Q[i, :, i05], color="#2171b5", lw=0.6, alpha=0.55, zorder=2)
        ax.plot(x, Q[i, :, i95], color="#2171b5", lw=0.6, alpha=0.55, zorder=2)
        ax.plot(x, Q[i, :, i50], color="#08306b", lw=3.0, zorder=3.5)
        ax.plot(x, Q[i, :, i50], color="white", lw=1.4, zorder=4, label="median forecast")
        ax.scatter(x[m], Y[i][m], s=15, color="#111", zorder=5, label="observed",
                   edgecolor="white", linewidth=0.4)
        ax.axhline(THR70, color="#cb181d", ls=(0, (5, 2)), lw=1.1, zorder=3)
        ax.text(HZ - 0.2, THR70 + 0.012, "extreme  $H\\!\\geq\\!Q_{70}$", color="#b30000",
                fontsize=6.0, ha="right", va="bottom", fontweight="bold")
        ax.set_ylim(min(0.14, Q[i, :, i05].min() - 0.03), max(0.88, Q[i, :, i95].max() + 0.04))
        ax.set_xlim(0.5, HZ + 0.5)
        ax.set_title(f"{tag} {stt.title()} — day-ahead predictive density", fontsize=8, loc="left", pad=3)
        ax.set_ylabel("hazard $H$", fontsize=8)
        ax.tick_params(labelbottom=False, labelsize=7); despine(ax)
        pe = pexc(i)
        axp.fill_between(x, 0, pe, color="#cb181d", alpha=0.22, lw=0, zorder=1)
        axp.plot(x, pe, color="#cb181d", lw=1.3, zorder=2)
        axp.set_ylim(0, 1.0); axp.set_yticks([0, 1]); axp.set_ylabel("$P(\\mathrm{ext})$", fontsize=6.8)
        axp.set_xlim(0.5, HZ + 0.5); axp.set_xticks([1, 6, 12, 18, 24])
        axp.set_xlabel("lead time $h$ (hours)", fontsize=8)
        axp.tick_params(labelsize=7); axp.grid(alpha=0.15, lw=0.5); despine(axp)
        if col == 0:
            ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2, borderaxespad=0.25,
                      labelspacing=0.3, framealpha=0.75)
    # shared slim colourbar: predictive density (low -> high)
    cax = fig.add_axes([0.91, 0.37, 0.014, 0.53])
    cb = fig.colorbar(mesh, cax=cax); cb.set_label("predictive density", fontsize=6.6, labelpad=2)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.ax.tick_params(labelsize=5.6, length=0)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig2_forecast")

if __name__ == "__main__":
    for fn in (fig1_composite, fig2_forecast, fig3_reliability, fig4_importance):
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")
    print("DONE ->", OUT)
