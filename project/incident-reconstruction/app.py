"""Streamlit dashboard: AI-Based Incident-Time Person Reconstruction.

Run:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database.connection import ping
from database.db_operations import DBOperations
from modules.path_manager import PathManager
from modules.travel_time import TravelTimeModel
from modules.timeline_generator import TimelineGenerator
from modules.reconstruction_engine import ReconstructionEngine
from modules.threat_engine import ThreatEngine
from modules.post_incident_tracker import PostIncidentTracker
from modules.evaluation import Evaluator
from modules.video_processor import VideoProcessor
from data.synthetic_data import SyntheticDataGenerator
from utils.config import THRESHOLD_HIGH, THRESHOLD_MEDIUM, classify
from utils.helpers import isoformat, parse_timestamp

st.set_page_config(
    page_title="Incident-Time Person Reconstruction",
    page_icon="🕵️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Component factory (cached in session state across reruns)                   #
# --------------------------------------------------------------------------- #
def init_components():
    if "components" in st.session_state:
        return st.session_state["components"]
    try:
        db_ops = DBOperations()  # pings MongoDB on construction
        pm = PathManager(db_ops=db_ops)
        if not pm.is_loaded():
            pm.load_from_db()
        comps = {
            "db": db_ops,
            "path_manager": pm,
            "travel_model": TravelTimeModel(db_ops=db_ops, path_manager=pm),
            "timeline": TimelineGenerator(db_ops=db_ops, path_manager=pm),
            "engine": ReconstructionEngine(db_ops=db_ops, path_manager=pm),
            "threat": ThreatEngine(),
            "post_incident": PostIncidentTracker(db_ops=db_ops, path_manager=pm),
            "evaluator": Evaluator(db_ops=db_ops),
            "processor": VideoProcessor(db_ops=db_ops),
            "synthetic": SyntheticDataGenerator(db_ops=db_ops, path_manager=pm),
            "connected": True,
        }
    except Exception as exc:  # MongoDB not running -> degraded UI
        comps = {"connected": False, "error": str(exc)}
    st.session_state["components"] = comps
    return comps


def require_db(comps):
    if not comps.get("connected"):
        st.error(
            "MongoDB is not reachable. Start MongoDB locally, e.g.:\n"
            "`\"C:\\Program Files\\MongoDB\\Server\\8.0\\bin\\mongod.exe\"` "
            "`--dbpath C:\\data\\db`\n\n"
            f"Details: {comps.get('error')}"
        )
        return None
    return comps["db"]


def get_best_person_image(db, person_id: str):
    """Return the image path of the highest-confidence observation for a person.

    Images are cropped person snapshots captured during video processing
    (see :mod:`modules.video_processor`). Returns None when no image exists.
    """
    try:
        obs = db.observations_for_person(person_id)
    except Exception:
        return None
    with_img = [
        o for o in obs
        if o.get("image_path") and Path(o["image_path"]).exists()
    ]
    if not with_img:
        return None
    # Resolve to an absolute path so st.image() works no matter what the
    # current working directory is when Streamlit is launched.
    return str(Path(
        max(with_img, key=lambda o: o.get("confidence") or 0)["image_path"]
    ).resolve())


def play_video(video_file: Path, key: str | None = None) -> None:
    """Play a local MP4 video robustly, with a download fallback.

    OpenCV writes processed videos with the FMP4 (MPEG-4 Part 2) codec, which
    browsers cannot decode — they need H.264/avc1. So st.video() may show a
    player but stay black. We read the file as bytes (more reliable than a
    path on Windows) and always offer a download button so the user can play
    the file locally in VLC, QuickTime, etc.
    """
    if not video_file.exists():
        st.error(f"Video file not found: {video_file}")
        return
    data = video_file.read_bytes()
    try:
        st.video(data)
    except Exception as exc:
        st.error(f"Browser cannot play this video ({video_file.name}): {exc}")
    st.download_button(
        label=f"⬇️ Download {video_file.name}",
        data=data,
        file_name=video_file.name,
        mime="video/mp4",
        key=key or f"dl_{video_file.name}",
    )


# --------------------------------------------------------------------------- #
# Plotly helpers                                                              #
# --------------------------------------------------------------------------- #
def network_figure(pm, highlighted_path: str | None = None) -> go.Figure:
    """Path network graph: green observable / red unobservable."""
    graph = pm.graph
    pos = pm.network_layout()
    fig = go.Figure()

    # Edges.
    for a, b in graph.edges():
        xa, ya = pos[a]
        xb, yb = pos[b]
        fig.add_trace(
            go.Scatter(
                x=[xa, xb, None],
                y=[ya, yb, None],
                mode="lines",
                line=dict(color="#94a3b8", width=3),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Nodes.
    for node in graph.nodes():
        x, y = pos[node]
        data = graph.nodes[node]
        observable = bool(data.get("is_observable", True))
        color = "#22c55e" if observable else "#ef4444"
        size = 26 if node == highlighted_path else 18
        line_width = 4 if node == highlighted_path else 1
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[node],
                textposition="bottom center",
                textfont=dict(size=11, color="#333"),
                marker=dict(
                    color=color,
                    size=size,
                    line=dict(width=line_width, color="#111"),
                    symbol="diamond" if data.get("camera_ids") else "circle",
                ),
                hovertemplate=(
                    f"<b>{node}</b><br>"
                    f"{'Observable' if observable else 'UNOBSERVABLE'}<br>"
                    f"Cameras: {len(data.get('camera_ids') or [])}"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=f"Affected path: {highlighted_path or '—'}  "
              "(green = observable, red = unobservable, diamond = camera)",
        showlegend=False,
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="#ffffff",
    )
    return fig


def trajectory_figure(events: list[dict], incident_time=None,
                      affected_path: str | None = None,
                      predicted_entry=None, predicted_exit=None) -> go.Figure:
    """Timeline of a person's path presence: observed vs inferred."""
    if not events:
        return go.Figure()

    # Order paths by first appearance so the y-axis follows the walk.
    order = []
    for e in events:
        p = e.get("path_id")
        if p and p not in order:
            order.append(p)
    y_index = {p: i for i, p in enumerate(order)}

    fig = go.Figure()
    for source, color, symbol, name in (
        ("camera", "#2563eb", "circle", "Observed (camera)"),
        ("inferred", "#ea580c", "x", "Inferred (reconstruction)"),
    ):
        xs, ys, labels = [], [], []
        for e in events:
            if e.get("source") != source or not e.get("timestamp"):
                continue
            xs.append(e["timestamp"])
            ys.append(y_index[e["path_id"]])
            labels.append(
                f"{e['path_id']}<br>{e.get('timestamp_iso','')}<br>{name}"
            )
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="markers+lines", name=name,
                    marker=dict(color=color, symbol=symbol, size=10),
                    line=dict(color=color, width=2, dash=("solid" if source == "camera" else "dot")),
                    text=labels, hoverinfo="text",
                )
            )

    ts = parse_timestamp
    if incident_time:
        incident_time = ts(incident_time)
        fig.add_vline(
            x=incident_time, line=dict(color="#dc2626", width=2, dash="dash"),
            annotation_text="Incident", annotation_position="top left",
        )
    if affected_path and affected_path in y_index:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=min([e["timestamp"] for e in events if e["timestamp"]] or [0]),
            x1=max([e["timestamp"] for e in events if e["timestamp"]] or [1]),
            y0=y_index[affected_path] - 0.4, y1=y_index[affected_path] + 0.4,
            fillcolor="#fbcfe8", opacity=0.3, line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=min([e["timestamp"] for e in events if e["timestamp"]] or [0]),
            y=y_index[affected_path],
            text="Affected path", showarrow=False,
            xanchor="left", font=dict(color="#9b1c1c", size=10),
        )
    if predicted_entry:
        fig.add_vline(x=ts(predicted_entry), line=dict(color="#16a34a", width=1, dash="dot"),
                      annotation_text="Est. entry", annotation_position="top right")
    if predicted_exit:
        fig.add_vline(x=ts(predicted_exit), line=dict(color="#16a34a", width=1, dash="dot"),
                      annotation_text="Est. exit", annotation_position="bottom right")

    fig.update_yaxes(
        tickmode="array", tickvals=list(y_index.values()),
        ticktext=list(y_index.keys()), title_text="Path",
    )
    fig.update_xaxes(title_text="Time (UTC)")
    fig.update_layout(
        height=420,
        margin=dict(l=60, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# --------------------------------------------------------------------------- #
# Pages                                                                       #
# --------------------------------------------------------------------------- #
def page_overview(db, pm):
    st.header("Overview Dashboard")
    try:
        persons = db.list_persons()
        cameras = db.list_cameras()
        paths = db.list_paths()
        incidents = db.list_incidents()
    except Exception as exc:
        st.error(f"Failed to load overview: {exc}")
        return

    observable = sum(1 for p in paths if p.get("is_observable"))
    unobservable = len(paths) - observable

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("People detected", len(persons))
    c2.metric("Cameras", len(cameras))
    c3.metric("Paths", len(paths))
    c4.metric("Observable paths", observable)
    c5.metric("Unobservable paths", unobservable)
    c6.metric("Incidents", len(incidents))

    st.subheader("Path observability")
    fig = go.Figure(
        data=[go.Pie(
            labels=["Observable", "Unobservable"],
            values=[observable, unobservable],
            marker=dict(colors=["#22c55e", "#ef4444"]),
            hole=0.4,
        )]
    )
    fig.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent incidents")
    if incidents:
        inc_df = pd.DataFrame([
            {
                "Incident ID": i["incident_id"],
                "Affected Path": i["affected_path_id"],
                "Time": i["incident_time"],
                "Severity": i["severity"],
                "Status": i.get("status", "open"),
            }
            for i in incidents[:10]
        ])
        st.dataframe(inc_df, use_container_width=True, hide_index=True)

        latest = incidents[0]
        preds = db.list_predictions(latest["incident_id"])
        if preds:
            st.subheader(f"Latest incident — {latest['incident_id']} "
                         f"({latest['affected_path_id']})")
            from utils.config import THRESHOLD_HIGH, THRESHOLD_MEDIUM
            bands = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            for p in preds:
                bands[p.get("risk_level", "LOW")] += 1
            b1, b2, b3 = st.columns(3)
            b1.metric("High probability", bands["HIGH"])
            b2.metric("Medium probability", bands["MEDIUM"])
            b3.metric("Low probability", bands["LOW"])
            fig2 = go.Figure(
                data=[go.Bar(
                    x=list(bands.keys()), y=list(bands.values()),
                    marker=dict(color=["#ef4444", "#f59e0b", "#22c55e"]),
                )]
            )
            fig2.update_layout(
                height=300, margin=dict(t=10, b=20),
                yaxis_title="People", xaxis_title="Risk level",
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No incidents registered yet. Use Generate Demo Dataset or "
                "register an incident.")


def page_camera_management(db, pm, processor):
    st.header("Camera Management")
    sub = st.tabs(["Cameras", "Upload & Process Video"])

    with sub[0]:
        st.subheader("Cameras")
        try:
            cameras = db.list_cameras()
            paths = db.list_paths()
        except Exception as exc:
            st.error(f"Failed to load cameras: {exc}")
            return

        if cameras:
            cam_df = pd.DataFrame([
                {
                    "Camera ID": c["camera_id"],
                    "Name": c["camera_name"],
                    "Path": c["path_id"],
                    "Location": c.get("location", ""),
                    "Status": c.get("status", "active"),
                }
                for c in cameras
            ])
            st.dataframe(cam_df, use_container_width=True, hide_index=True)

            st.subheader("Delete camera")
            to_del = st.selectbox("Select camera", [""] + [c["camera_id"] for c in cameras])
            if to_del and st.button("Delete camera", type="secondary"):
                try:
                    ok = db.delete_camera(to_del)
                    st.success("Deleted.") if ok else st.warning("Not found.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")

        st.divider()
        st.subheader("Add / edit camera")
        path_options = {p["path_id"]: p["path_id"] for p in paths}
        col1, col2 = st.columns(2)
        with col1:
            cam_id = st.text_input("Camera ID", key="cam_new_id",
                                   placeholder="e.g. CAM_P4")
            cam_name = st.text_input("Camera name", key="cam_new_name")
            cam_path = st.selectbox("Assign to path",
                                    options=[""] + list(path_options.keys()),
                                    key="cam_new_path")
        with col2:
            cam_loc = st.text_input("Location", key="cam_new_loc")
            cam_status = st.selectbox("Status", ["active", "maintenance", "offline"],
                                      key="cam_new_status")
        if st.button("Save camera", type="primary"):
            if not cam_id or not cam_path:
                st.error("Camera ID and path are required.")
            else:
                try:
                    if db.get_camera(cam_id):
                        db.update_camera(cam_id, {
                            "camera_name": cam_name,
                            "path_id": cam_path,
                            "location": cam_loc,
                            "status": cam_status,
                        })
                        pm.assign_camera(cam_path, cam_id)
                        st.success(f"Camera {cam_id} updated.")
                    else:
                        db.insert_camera(cam_id, cam_name, cam_path, cam_loc, cam_status)
                        pm.assign_camera(cam_path, cam_id)
                        st.success(f"Camera {cam_id} added.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Save failed: {exc}")

    with sub[1]:
        st.subheader("Upload CCTV footage (MP4)")
        try:
            cameras = db.list_cameras()
        except Exception as exc:
            st.error(f"Failed to load cameras: {exc}")
            return

        if not cameras:
            st.info("Add a camera first (Cameras tab).")
            return

        cam_map = {f"{c['camera_id']} (path {c['path_id']})": c for c in cameras}
        choice = st.selectbox("Camera / path", options=list(cam_map.keys()))
        camera = cam_map[choice]

        uploaded = st.file_uploader("Upload MP4 video", type=["mp4"])
        c_date, c_time, c_sec = st.columns([2, 2, 1])
        with c_date:
            start_date = st.date_input("Recording date", value=date.today())
        with c_time:
            start_time = st.time_input("Recording start time", value=time(8, 0))
        with c_sec:
            start_seconds = st.number_input("Seconds", min_value=0, max_value=59,
                                            value=0, step=1,
                                            help="Seconds for precise timestamp")
        start_time = start_time.replace(second=start_seconds)
        frame_skip = st.slider("Process every Nth frame", 1, 10, 1,
                               help="Higher = faster, fewer observations")
        max_frames = st.number_input("Max frames (0 = all)", min_value=0, value=0)

        if uploaded and st.button("Process video", type="primary"):
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                out_name = f"processed_{camera['camera_id']}.mp4"
                out_path = str((Path("data/processed") / out_name).resolve())
                start_dt = datetime.combine(start_date, start_time)

                progress = st.progress(0)
                status = st.empty()

                def cb(cur, total):
                    pct = cur / total if total else cur / 100
                    progress.progress(min(pct, 1.0))
                    status.text(f"Frame {cur}/{total}...")

                summary = processor.process_video(
                    tmp_path, camera["camera_id"], camera["path_id"],
                    start_time=start_dt,
                    frame_skip=max(1, int(frame_skip)),
                    max_frames=int(max_frames) if max_frames else None,
                    output_path=out_path,
                    progress_callback=cb,
                )
                progress.empty(); status.empty()
                st.success(
                    f"Processed {summary['frames_processed']} frames, "
                    f"{summary['detections']} detections, "
                    f"{summary['observations_saved']} observations saved."
                )
                st.json(summary)
                if Path(out_path).exists():
                    play_video(Path(out_path), key=f"dl_new_{camera['camera_id']}")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")


def page_video_processing(db, pm, processor):
    st.header("Video Processing")
    st.info(
        "Processing runs YOLO person detection + ByteTrack tracking on every "
        "frame and saves camera observations to MongoDB. Tracking IDs are local "
        "to each camera; associate them with global Person IDs on the Person "
        "Tracking page."
    )
    processed_dir = Path("data/processed").resolve()
    files = sorted(processed_dir.glob("*.mp4")) if processed_dir.exists() else []
    if files:
        st.subheader("Previously processed videos")
        for f in files:
            st.write(f.name)
            play_video(f, key=f"dl_existing_{f.name}")


def page_path_network(db, pm):
    st.header("Path Network")
    tabs = st.tabs(["Network view", "Manage paths", "Connections & cameras"])

    with tabs[0]:
        st.subheader("Connected path network")
        affected = st.selectbox(
            "Highlight affected path",
            options=[""] + [p["path_id"] for p in pm.list_paths()],
        )
        fig = network_figure(pm, highlighted_path=affected or None)
        st.plotly_chart(fig, use_container_width=True)

        paths = pm.list_paths()
        if paths:
            net_df = pd.DataFrame([
                {
                    "Path ID": p["path_id"],
                    "Name": p.get("name", ""),
                    "Length (m)": p.get("length"),
                    "Avg speed (m/min)": p.get("average_speed"),
                    "Observable": "Yes" if p.get("is_observable") else "No",
                    "Cameras": len(p.get("camera_ids") or []),
                    "Connected": ", ".join(p.get("connected_paths") or []),
                }
                for p in paths
            ])
            st.dataframe(net_df, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Add / edit path")
        col1, col2 = st.columns(2)
        with col1:
            p_id = st.text_input("Path ID", key="p_new_id",
                                 placeholder="e.g. P7")
            p_name = st.text_input("Name", key="p_new_name")
            p_len = st.number_input("Length (meters)", min_value=1.0, value=100.0)
        with col2:
            p_speed = st.number_input("Avg walking speed (m/min)",
                                      min_value=1.0, value=84.0)
            p_obs = st.selectbox("Observable", ["Yes", "No"], key="p_new_obs")
        if st.button("Save path", type="primary"):
            try:
                pm.add_path(
                    p_id, p_name or p_id, float(p_len), float(p_speed),
                    is_observable=(p_obs == "Yes"),
                )
                st.success(f"Path {p_id} added.")
                st.rerun()
            except Exception as exc:
                st.error(f"Add failed: {exc}")

        st.divider()
        st.subheader("Edit / delete existing path")
        paths = pm.list_paths()
        if paths:
            target = st.selectbox("Path", options=[p["path_id"] for p in paths],
                                  key="p_edit_target")
            cur = pm.get_path(target)
            e1, e2 = st.columns(2)
            with e1:
                e_name = st.text_input("Name", value=cur.get("name", ""),
                                       key="p_edit_name")
                e_len = st.number_input("Length (m)", value=float(cur.get("length", 100)),
                                        key="p_edit_len")
            with e2:
                e_speed = st.number_input("Avg speed (m/min)",
                                          value=float(cur.get("average_speed", 84)),
                                          key="p_edit_speed")
                e_obs = st.selectbox("Observable",
                                     ["Yes", "No"],
                                     index=0 if cur.get("is_observable") else 1,
                                     key="p_edit_obs")
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.button("Update path"):
                    try:
                        pm.update_path(
                            target, name=e_name or target, length=float(e_len),
                            average_speed=float(e_speed),
                            is_observable=(e_obs == "Yes"),
                        )
                        st.success("Updated.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Update failed: {exc}")
            with ec2:
                if st.button("Delete path", type="secondary"):
                    try:
                        pm.delete_path(target)
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")

    with tabs[2]:
        st.subheader("Connect / disconnect paths")
        paths = pm.list_paths()
        path_ids = [p["path_id"] for p in paths]
        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox("Path A", options=path_ids, key="conn_a")
            b = st.selectbox("Path B", options=path_ids, key="conn_b")
            if st.button("Connect"):
                try:
                    pm.connect_paths(a, b)
                    st.success(f"Connected {a} <-> {b}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Connect failed: {exc}")
        with c2:
            if st.button("Disconnect"):
                try:
                    pm.disconnect_paths(a, b)
                    st.success(f"Disconnected {a} <-> {b}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Disconnect failed: {exc}")

        st.divider()
        st.subheader("Assign camera to path")
        cameras = db.list_cameras()
        cam_ids = [c["camera_id"] for c in cameras]
        m1, m2 = st.columns(2)
        with m1:
            ap = st.selectbox("Path", options=path_ids, key="assign_path")
            ac = st.selectbox("Camera", options=[""] + cam_ids, key="assign_cam")
        with m2:
            if st.button("Assign"):
                try:
                    pm.assign_camera(ap, ac)
                    st.success(f"Assigned {ac} to {ap}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Assign failed: {exc}")
            rc = st.selectbox("Remove camera from path",
                              options=[""] + cam_ids, key="remove_cam")
            rp = st.selectbox("From path", options=path_ids, key="remove_path")
            if st.button("Remove"):
                try:
                    pm.unassign_camera(rp, rc)
                    st.success(f"Removed {rc} from {rp}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Remove failed: {exc}")


def page_person_tracking(db, timeline_gen):
    st.header("Person Tracking")

    # ------------------------------------------------------------------ #
    # Register a new person                                              #
    # ------------------------------------------------------------------ #
    st.subheader("Register person")
    with st.form("register_person_form"):
        new_pid = st.text_input("Person ID", placeholder="e.g. Person_001",
                                help="Anonymous ID — no real names or photos stored.")
        reg_submitted = st.form_submit_button("Create person", type="primary")
    if reg_submitted:
        if not new_pid:
            st.error("Person ID is required.")
        else:
            try:
                db.insert_person(new_pid)
                st.success(f"Person {new_pid} created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Create failed: {exc}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Link video detections to persons                                   #
    # ------------------------------------------------------------------ #
    st.subheader("Link video detections to persons")
    st.caption(
        "After processing a video, detections appear here with no person ID. "
        "Pick a person for each detection so reconstruction can evaluate them."
    )
    unassigned = db.list_unassigned_observations()
    persons = db.list_persons()
    person_options = [p["person_id"] for p in persons]

    if not unassigned:
        st.info("No unassigned video detections. Process a video on Camera Management first.")
    elif not person_options:
        st.info("No persons registered yet. Create one above, then link detections.")
    else:
        for obs in unassigned[:50]:  # cap to keep the UI snappy
            oid = obs.get("id")
            cols = st.columns([2, 2, 2, 2, 2])
            cols[0].write(f"{obs.get('camera_id')} — tracking {obs.get('tracking_id')}")
            cols[1].write(obs.get("path_id") or "—")
            cols[2].write(str(obs.get("timestamp")) or "—")
            cols[3].write(f"conf {obs.get('confidence', 0):.2f}" if obs.get("confidence") is not None else "—")
            selected = cols[4].selectbox(
                "Assign to",
                options=[""] + person_options,
                key=f"assign_{oid}",
                label_visibility="collapsed",
            )
            if selected and st.button("Link", key=f"link_{oid}"):
                try:
                    db.update_observation_person(oid, selected)
                    st.success(f"Linked to {selected}.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Link failed: {exc}")

    st.divider()

    # ------------------------------------------------------------------ #
    # Existing person timeline                                           #
    # ------------------------------------------------------------------ #
    if not persons:
        st.info("No persons in the database. Register one above or link a video detection.")
        return

    person_id = st.selectbox("Select person", options=[p["person_id"] for p in persons])
    if not person_id:
        return

    # Person's captured image (if any video has been processed for them).
    img_path = get_best_person_image(db, person_id)
    if img_path:
        st.image(img_path, caption=f"{person_id} — captured person image", width=220)
    else:
        st.caption("No captured image available. Process a video containing this person first.")

    try:
        tl = timeline_gen.timeline(person_id)
    except Exception as exc:
        st.error(f"Timeline load failed: {exc}")
        return

    st.subheader(f"Timeline — {person_id}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", tl["event_count"])
    m2.metric("Observed (camera)", tl["observed_count"])
    m3.metric("Inferred", tl["inferred_count"])
    m4.metric("First observed", tl["first_observed"] or "—")

    st.write(f"Last observed: {tl['last_observed'] or '—'}")
    st.write("Reconstructed trajectory: "
             + " → ".join(tl["path_history"]) if tl["path_history"] else "—")

    if tl["events"]:
        ev_df = pd.DataFrame([
            {
                "Time": e.get("timestamp_iso"),
                "Path": e.get("path_id"),
                "Camera": e.get("camera_id") or "—",
                "Source": e.get("source", "camera").title(),
                "Confidence": e.get("confidence"),
            }
            for e in tl["events"]
        ])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

    if tl["reconstructed_trajectory"]:
        st.subheader("Trajectory visualization")
        fig = trajectory_figure(tl["reconstructed_trajectory"])
        st.plotly_chart(fig, use_container_width=True)

    if tl["inferred_transitions"]:
        st.subheader("Inferred path transitions")
        inf_df = pd.DataFrame([
            {
                "Time": e.get("timestamp_iso"),
                "Path": e.get("path_id"),
                "Source": e.get("source"),
            }
            for e in tl["inferred_transitions"]
        ])
        st.dataframe(inf_df, use_container_width=True, hide_index=True)
        st.caption(
            "Inferred transitions are NOT camera observations — they are the "
            "shortest-path fill-ins between consecutive observed paths."
        )


def page_incident_management(db, pm, engine, threat, post_incident, synthetic):
    st.header("Incident Management")

    st.subheader("Register incident")
    paths = pm.list_paths()
    with st.form("incident_form"):
        inc_id = st.text_input("Incident ID", placeholder="e.g. INC_001")
        aff_path = st.selectbox("Affected path",
                                options=[p["path_id"] for p in paths])
        c_date, c_time, c_sec = st.columns([2, 2, 1])
        with c_date:
            inc_date = st.date_input("Date", value=date.today())
        with c_time:
            inc_time = st.time_input("Time (UTC)", value=time(10, 30))
        with c_sec:
            inc_seconds = st.number_input("Seconds", min_value=0, max_value=59,
                                          value=0, step=1,
                                          help="Seconds for precise incident time")
        inc_time = inc_time.replace(second=inc_seconds)
        severity = st.selectbox("Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
        desc = st.text_area("Description", placeholder="What happened?")
        submitted = st.form_submit_button("Register incident", type="primary")
    if submitted:
        if not inc_id:
            st.error("Incident ID is required.")
        else:
            try:
                incident_dt = datetime.combine(inc_date, inc_time)
                db.insert_incident(inc_id, aff_path, incident_dt, severity, desc)
                st.success(f"Incident {inc_id} registered.")
                st.session_state["trigger_reconstruction"] = inc_id
                st.rerun()
            except Exception as exc:
                st.error(f"Register failed: {exc}")

    if st.session_state.get("trigger_reconstruction"):
        inc_id = st.session_state.pop("trigger_reconstruction")
        inc = db.get_incident(inc_id)
        if inc:
            st.info(f"Auto-reconstructing for {inc_id} on {inc['affected_path_id']} ...")
            try:
                results = engine.reconstruct_for_incident(
                    inc_id, inc["affected_path_id"],
                    parse_timestamp(inc["incident_time"]),
                )
                bands = threat.classify_results(results)
                for r in results:
                    post_incident.analyze(
                        r["person_id"], parse_timestamp(inc["incident_time"])
                    )
                st.success(
                    f"Reconstruction complete: {len(results)} people evaluated "
                    f"({len(bands['HIGH'])} HIGH, {len(bands['MEDIUM'])} MEDIUM, "
                    f"{len(bands['LOW'])} LOW)."
                )
            except Exception as exc:
                st.error(f"Reconstruction failed: {exc}")

    st.divider()
    st.subheader("Incident history")
    try:
        incidents = db.list_incidents()
    except Exception as exc:
        st.error(f"Failed to load incidents: {exc}")
        return
    if incidents:
        inc_df = pd.DataFrame([
            {
                "Incident ID": i["incident_id"],
                "Affected Path": i["affected_path_id"],
                "Time": i["incident_time"],
                "Severity": i["severity"],
                "Status": i.get("status", "open"),
                "Created": i.get("created_at"),
            }
            for i in incidents
        ])
        st.dataframe(inc_df, use_container_width=True, hide_index=True)

        st.subheader("Trigger reconstruction for an incident")
        sel = st.selectbox("Incident",
                           options=[""] + [i["incident_id"] for i in incidents])
        if sel and st.button("Run reconstruction now", type="primary"):
            inc = db.get_incident(sel)
            try:
                results = engine.reconstruct_for_incident(
                    sel, inc["affected_path_id"], parse_timestamp(inc["incident_time"]),
                )
                bands = threat.classify_results(results)
                for r in results:
                    post_incident.analyze(
                        r["person_id"], parse_timestamp(inc["incident_time"])
                    )
                st.success(
                    f"Done: {len(results)} people "
                    f"({len(bands['HIGH'])} HIGH / {len(bands['MEDIUM'])} MEDIUM / "
                    f"{len(bands['LOW'])} LOW)."
                )
            except Exception as exc:
                st.error(f"Reconstruction failed: {exc}")
    else:
        st.info("No incidents yet. Register one above or run Generate Demo Dataset.")


def page_affected_person_analysis(db, pm, engine, threat):
    st.header("Affected-Person Analysis")

    incidents = db.list_incidents()
    if not incidents:
        st.info("No incidents registered yet.")
        return

    inc_id = st.selectbox("Incident",
                          options=[i["incident_id"] for i in incidents])
    inc = db.get_incident(inc_id)
    st.write(f"Affected path: **{inc['affected_path_id']}**  |  "
             f"Time: **{inc['incident_time']}**  |  "
             f"Severity: **{inc['severity']}**")

    st.subheader("Classification thresholds")
    c1, c2, c3 = st.columns([2, 2, 1])
    high_thr = c1.number_input("HIGH threshold (≥)", min_value=0.0,
                               max_value=100.0, value=float(threat.high),
                               step=1.0)
    med_thr = c2.number_input("MEDIUM threshold (≥)", min_value=0.0,
                              max_value=100.0, value=float(threat.medium),
                              step=1.0)
    if c3.button("Apply thresholds"):
        threat.update_thresholds(high_thr, med_thr)
        st.success("Thresholds updated.")

    try:
        predictions = db.list_predictions(inc_id)
    except Exception as exc:
        st.error(f"Failed to load predictions: {exc}")
        return

    if not predictions:
        st.info("No predictions yet. Run reconstruction first.")
        return

    # Re-classify with current thresholds (in-memory + persist band).
    bands = threat.classify_results(predictions)
    for r in predictions:
        db.update_one(
            "predictions",
            {"incident_id": inc_id, "person_id": r["person_id"]},
            {"risk_level": r["risk_level"]},
        )

    st.subheader("🚨 Threat gallery — persons with image & probability")
    st.caption(
        "Each card shows the person's captured image (from processed video) and "
        "their probability of being on the affected path at the incident time. "
        "Higher probability = more likely to have been in the unobservable path."
    )
    # Build the ranked prediction dataframe.
    all_rows = []
    for r in predictions:
        all_rows.append({
            "Person ID": r["person_id"],
            "Probability": round(float(r["probability"]), 1),
            "Classification": r.get("risk_level", "LOW"),
            "Entry": isoformat(r.get("predicted_entry_time")),
            "Exit": isoformat(r.get("predicted_exit_time")),
            "Route": " → ".join(r.get("reconstructed_path") or []),
            "Last observed": r.get("last_observed_location") or "—",
            "Explanation": r.get("explanation", ""),
        })
    df = pd.DataFrame(all_rows).sort_values("Probability", ascending=False)

    threat_df = df[df["Classification"].isin(["HIGH", "MEDIUM"])].copy()
    if not threat_df.empty:
        # Show 3 cards per row.
        for start in range(0, len(threat_df), 3):
            chunk = threat_df.iloc[start:start + 3]
            cols = st.columns(len(chunk))
            for col, (_, row) in zip(cols, chunk.iterrows()):
                with col:
                    pid = row["Person ID"]
                    prob = float(row["Probability"])
                    band = row["Classification"]
                    band_color = "#ef4444" if band == "HIGH" else "#f59e0b"
                    img_path = get_best_person_image(db, pid)
                    if img_path:
                        st.image(img_path, caption=pid, width=200)
                    else:
                        st.info(f"{pid}\nNo image", width=200)
                    st.markdown(
                        f"<div style='text-align:center; margin-top:4px;'>"
                        f"<strong>{pid}</strong><br>"
                        f"<span style='color:{band_color}; font-weight:bold; font-size:1.1em;'>"
                        f"{band}</span><br>"
                        f"<span style='font-size:1.3em;'><strong>{prob:.1f}%</strong></span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        f"Entry: {row['Entry'] or '—'} | "
                        f"Exit: {row['Exit'] or '—'} | "
                        f"Route: {row['Route'] or '—'}"
                    )
    else:
        st.info("No HIGH or MEDIUM threat persons for this incident.")

    st.divider()

    # ------------------------------------------------------------------ #
    # Ranked table                                                       #
    # ------------------------------------------------------------------ #
    st.subheader("Affected-person list (ranked by probability)")
    st.dataframe(
        df[["Person ID", "Probability", "Classification", "Entry", "Exit",
            "Route", "Last observed"]],
        use_container_width=True, hide_index=True,
    )

    # ------------------------------------------------------------------ #
    # Detail: image + probability side by side                           #
    # ------------------------------------------------------------------ #
    st.subheader("Prediction details")
    detail_pid = st.selectbox("Person", options=[""] + df["Person ID"].tolist(),
                              key="detail_person")
    if detail_pid:
        row = df[df["Person ID"] == detail_pid].iloc[0]
        prob = float(row["Probability"])
        band = row["Classification"]
        band_color = "#ef4444" if band == "HIGH" else ("#f59e0b" if band == "MEDIUM" else "#22c55e")
        img_path = get_best_person_image(db, detail_pid)
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            if img_path:
                st.image(img_path, caption=f"{detail_pid}", width=260)
                st.markdown(
                    f"<div style='text-align:center; margin-top:6px;'>"
                    f"<strong>Probability of being on affected path:</strong> "
                    f"<span style='color:{band_color}; font-size:1.5em; font-weight:bold;'>"
                    f"{prob:.1f}%</span><br>"
                    f"<strong>Classification:</strong> {band}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(f"No captured image available for {detail_pid}.")
                st.markdown(
                    f"<div style='text-align:center; margin-top:6px;'>"
                    f"<strong>Probability:</strong> {prob:.1f}%  |  "
                    f"<strong>Classification:</strong> {band}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        with col_txt:
            st.write(row["Explanation"])
            st.write(
                f"**Estimated entry:** {row['Entry'] or '—'}  |  "
                f"**Estimated exit:** {row['Exit'] or '—'}"
            )
            st.write(f"**Reconstructed route:** {row['Route'] or '—'}")


def page_movement_reconstruction(db, pm, engine, post_incident):
    st.header("Movement Reconstruction")

    incidents = db.list_incidents()
    if not incidents:
        st.info("No incidents registered yet.")
        return

    inc_id = st.selectbox("Incident",
                          options=[i["incident_id"] for i in incidents],
                          key="rec_incident")
    inc = db.get_incident(inc_id)
    affected = inc["affected_path_id"]
    inc_time = parse_timestamp(inc["incident_time"])

    predictions = db.list_predictions(inc_id)
    if not predictions:
        st.info("Run reconstruction for this incident first.")
        return

    person_id = st.selectbox("Person",
                             options=[p["person_id"] for p in predictions],
                             key="rec_person")
    result = next((p for p in predictions if p["person_id"] == person_id), None)
    if not result:
        return

    st.subheader(f"{person_id} — {result.get('risk_level', 'LOW')} "
                 f"({result.get('probability', 0):.1f}%)")
    st.write(result.get("explanation", ""))
    st.write(
        f"Estimated entry: {isoformat(result.get('predicted_entry_time')) or '—'}  |  "
        f"Estimated exit: {isoformat(result.get('predicted_exit_time')) or '—'}"
    )
    st.write(
        f"Reconstructed route: {' → '.join(result.get('reconstructed_path') or []) or '—'}"
    )

    # Timeline figure (pre + post incident).
    from modules.timeline_generator import TimelineGenerator
    tl = TimelineGenerator(db_ops=db, path_manager=pm).timeline(person_id)
    fig = trajectory_figure(
        tl["reconstructed_trajectory"],
        incident_time=inc["incident_time"],
        affected_path=affected,
        predicted_entry=result.get("predicted_entry_time"),
        predicted_exit=result.get("predicted_exit_time"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Post-incident status")
    post = post_incident.analyze(person_id, inc["incident_time"])
    st.write(post["post_incident_status"])
    st.write(
        f"Confirmed movement: **{post['confirmed']}**  |  "
        f"Last location: **{post['last_observed_location'] or '—'}**"
    )
    if post["path_history_after"]:
        st.write("Path after incident: " + " → ".join(
            post["path_history_after"]))
    st.caption(
        "Green dotted = observed after the incident; orange = inferred. "
        "This indicates potential exposure, not confirmed presence."
    )


def page_evaluation(db, evaluator):
    st.header("Evaluation")

    incidents = db.list_incidents()
    if not incidents:
        st.info("No incidents registered yet.")
        return

    inc_id = st.selectbox("Incident",
                          options=[i["incident_id"] for i in incidents],
                          key="eval_incident")
    inc = db.get_incident(inc_id)
    st.write(f"Affected path: **{inc['affected_path_id']}**  |  "
             f"Time: **{inc['incident_time']}**")
    st.caption(
        "Metrics compare predictions against the synthetic ground truth. "
        "Ground truth is never used to train or calibrate the algorithm."
    )

    try:
        metrics = evaluator.evaluate(inc_id)
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        return

    if "error" in metrics:
        st.warning(metrics["error"])
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{metrics['precision']*100:.1f}%")
    c2.metric("Recall", f"{metrics['recall']*100:.1f}%")
    c3.metric("F1 score", f"{metrics['f1']*100:.1f}%")
    c4.metric("Localization accuracy", f"{metrics['localization_accuracy']*100:.1f}%")
    c5.metric("Path reconstruction accuracy",
              f"{metrics['path_reconstruction_accuracy']*100:.1f}%")

    st.metric("Avg travel-time error (min)",
              f"{metrics['travel_time_error_minutes']:.1f}"
              if metrics["travel_time_error_minutes"] is not None else "—")

    st.subheader("Counts")
    ct = metrics["counts"]
    d1, d2, d3 = st.columns(3)
    d1.metric("True affected", ct["true_affected"])
    d2.metric("Predicted affected", ct["predicted_affected"])
    d3.metric("TP / FP / FN",
              f"{ct['tp']} / {ct['fp']} / {ct['fn']}")

    st.subheader("Per-person detail")
    if metrics["details"]:
        det = pd.DataFrame(metrics["details"])
        st.dataframe(det, use_container_width=True, hide_index=True)

    st.subheader("Metric definitions")
    st.markdown(
        "- **Precision**: of people predicted affected, fraction truly on the "
        "affected path at incident time.\n"
        "- **Recall**: of truly affected people, fraction predicted affected.\n"
        "- **F1**: harmonic mean of precision and recall.\n"
        "- **Incident-time localization accuracy**: fraction of predictions "
        "whose reconstructed position at the incident time equals the affected path.\n"
        "- **Path reconstruction accuracy**: fraction whose full reconstructed "
        "path sequence matches ground truth.\n"
        "- **Travel-time estimation error**: mean absolute difference between "
        "predicted and actual entry time (minutes)."
    )


def page_reports(db, engine, threat):
    st.header("Reports")

    incidents = db.list_incidents()
    if not incidents:
        st.info("No incidents yet.")
        return

    inc_id = st.selectbox("Incident",
                          options=[i["incident_id"] for i in incidents],
                          key="rep_incident")
    inc = db.get_incident(inc_id)
    predictions = db.list_predictions(inc_id)

    st.subheader("Incident summary")
    st.markdown(f"""
    - **Incident ID:** {inc['incident_id']}
    - **Affected path:** {inc['affected_path_id']}
    - **Time:** {inc['incident_time']}
    - **Severity:** {inc['severity']}
    - **People evaluated:** {len(predictions)}
    - **High:** {sum(1 for p in predictions if p.get('risk_level')=='HIGH')}
    - **Medium:** {sum(1 for p in predictions if p.get('risk_level')=='MEDIUM')}
    - **Low:** {sum(1 for p in predictions if p.get('risk_level')=='LOW')}
    """)

    st.divider()
    st.subheader("Export affected-person report (CSV)")
    if predictions:
        csv_df = pd.DataFrame([
            {
                "person_id": p["person_id"],
                "probability": p["probability"],
                "risk_level": p.get("risk_level"),
                "predicted_entry_time": p.get("predicted_entry_time"),
                "predicted_exit_time": p.get("predicted_exit_time"),
                "reconstructed_path": " → ".join(p.get("reconstructed_path") or []),
                "explanation": p.get("explanation"),
            }
            for p in predictions
        ])
        st.download_button(
            "Download predictions CSV",
            data=csv_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{inc_id}_predictions.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Export movement timelines (CSV)")
    try:
        observations = db.find_many("observations")
    except Exception as exc:
        st.error(f"Failed to load observations: {exc}")
        return
    if observations:
        obs_df = pd.DataFrame([
            {
                "person_id": o["person_id"],
                "camera_id": o["camera_id"],
                "path_id": o["path_id"],
                "timestamp": o.get("timestamp"),
                "tracking_id": o.get("tracking_id"),
                "confidence": o.get("confidence"),
                "source": o.get("source"),
            }
            for o in observations
        ])
        st.download_button(
            "Download observations CSV",
            data=obs_df.to_csv(index=False).encode("utf-8"),
            file_name="observations.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Historical incidents")
    hist = pd.DataFrame([
        {
            "Incident ID": i["incident_id"],
            "Path": i["affected_path_id"],
            "Time": i["incident_time"],
            "Severity": i["severity"],
            "Status": i.get("status", "open"),
        }
        for i in db.list_incidents()
    ])
    st.dataframe(hist, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# App shell                                                                   #
# --------------------------------------------------------------------------- #
def main():
    comps = init_components()
    if not comps.get("connected"):
        st.error(
            "MongoDB is not reachable. Start MongoDB locally, e.g.:\n"
            "`\"C:\\Program Files\\MongoDB\\Server\\8.0\\bin\\mongod.exe\"` "
            "`--dbpath C:\\data\\db`\n\n"
            f"Details: {comps.get('error')}"
        )
        st.stop()
    db = comps["db"]
    pm = comps["path_manager"]

    with st.sidebar:
        st.title("Incident Reconstruction")
        st.caption("MongoDB: connected ✅")
        st.divider()
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Camera Management",
                "Video Processing",
                "Path Network",
                "Person Tracking",
                "Incident Management",
                "Affected-Person Analysis",
                "Movement Reconstruction",
                "Evaluation",
                "Reports",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.subheader("Demo")
        st.caption("No video required — fully synthetic.")
        if st.button("Generate demo dataset", use_container_width=True):
            try:
                summary = comps["synthetic"].generate_demo()
                pm.refresh()
                st.session_state["demo_summary"] = summary
                st.success(
                    f"Demo dataset generated: {summary['persons']} people, "
                    f"{len(summary['cameras'])} cameras, incident "
                    f"{summary['incident_id']} on {summary['affected_path']} "
                    f"at {summary['incident_time']}."
                )
                st.write("Ground-truth affected: ",
                         ", ".join(summary["ground_truth_affected"]))
            except Exception as exc:
                st.error(f"Demo generation failed: {exc}")
        if st.button("Run demo incident", use_container_width=True):
            try:
                inc = db.get_incident("INC_001")
                if not inc:
                    st.error("No INC_001 — generate demo data first.")
                else:
                    results = comps["engine"].reconstruct_for_incident(
                        "INC_001", inc["affected_path_id"],
                        parse_timestamp(inc["incident_time"]),
                    )
                    comps["threat"].classify_results(results)
                    for r in results:
                        comps["post_incident"].analyze(
                            r["person_id"], parse_timestamp(inc["incident_time"])
                        )
                    bands = comps["threat"].summary(results)
                    st.success(
                        f"Demo reconstruction complete: {bands['counts']['HIGH']} HIGH, "
                        f"{bands['counts']['MEDIUM']} MEDIUM, "
                        f"{bands['counts']['LOW']} LOW."
                    )
            except Exception as exc:
                st.error(f"Demo incident failed: {exc}")

        st.divider()
        st.subheader("Reset")
        st.caption(
            "⚠️ Deletes ALL data — people, cameras, paths, observations, "
            "incidents, predictions, ground truth."
        )
        if st.button("Clear database", type="secondary", use_container_width=True):
            try:
                comps["synthetic"].clear_demo_data()
                pm.refresh()
                st.session_state.pop("demo_summary", None)
                st.session_state.pop("trigger_reconstruction", None)
                st.success("Database cleared — all data removed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")

    if page == "Overview":
        page_overview(db, pm)
    elif page == "Camera Management":
        page_camera_management(db, pm, comps["processor"])
    elif page == "Video Processing":
        page_video_processing(db, pm, comps["processor"])
    elif page == "Path Network":
        page_path_network(db, pm)
    elif page == "Person Tracking":
        page_person_tracking(db, comps["timeline"])
    elif page == "Incident Management":
        page_incident_management(db, pm, comps["engine"], comps["threat"],
                                 comps["post_incident"], comps["synthetic"])
    elif page == "Affected-Person Analysis":
        page_affected_person_analysis(db, pm, comps["engine"], comps["threat"])
    elif page == "Movement Reconstruction":
        page_movement_reconstruction(db, pm, comps["engine"], comps["post_incident"])
    elif page == "Evaluation":
        page_evaluation(db, comps["evaluator"])
    elif page == "Reports":
        page_reports(db, comps["engine"], comps["threat"])


if __name__ == "__main__":
    main()