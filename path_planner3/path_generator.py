#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS2 PathGenerator:
- /clicked_point (geometry_msgs/PointStamped) 로 제어점(cp) 수집 (4개 단위 → 3차 베지어 1세그먼트)
- /map (nav_msgs/OccupancyGrid) 구독해 장애물 정보를 원(인플레이트 반경)으로 변환
- 디카스텔쥬 분할 기반 충돌 구간 탐색 → 내부 제어점(1,2) 외측으로 살짝 밀어 안전거리 확보
- 최종 베지어 곡선을 샘플링하여 /path (nav_msgs/Path) 발행

의존:
  - rclpy, numpy
  - SciPy 없이 동작(간단한 반복-밀어내기 방식). 고급 QP 최적화로 바꾸고 싶으면 SciPy 추가 후 교체 가능.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Sequence
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # ★ QoS 추가

import numpy as np
import rclpy as rp
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.node import Node

# ------------------------------
# 타입 별칭
# ------------------------------
Point = Tuple[float, float]


# ------------------------------
# 베지어/기하 유틸
# ------------------------------
def lerp2(p: Point, q: Point, t: float) -> Point:
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def de_casteljau_split(ctrl: Sequence[Point], t: float):
    """3차 포함 임의 차수 베지어 분할: 왼쪽/오른쪽 제어점 반환"""
    work = [list(ctrl)]
    n = len(ctrl) - 1
    for _ in range(1, n + 1):
        prev = work[-1]
        cur = [lerp2(prev[i], prev[i + 1], t) for i in range(len(prev) - 1)]
        work.append(cur)
    left = [work[i][0] for i in range(n + 1)]
    right = [work[n - i][i] for i in range(n + 1)]
    return left, right


def bezier_eval(ctrl: Sequence[Point], t: float) -> Point:
    """디카스텔쥬로 점 평가"""
    pts = list(ctrl)
    n = len(pts) - 1
    for _ in range(1, n + 1):
        pts = [lerp2(pts[i], pts[i + 1], t) for i in range(len(pts) - 1)]
    return pts[0]


def bezier_flatness(ctrl: Sequence[Point]) -> float:
    """끝점 직선 대비 내부 CP 최대 편차(평탄성)"""
    p0, pn = ctrl[0], ctrl[-1]
    x0, y0 = p0
    x1, y1 = pn
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy)
    if denom == 0.0:
        return 0.0
    mx = 0.0
    for p in ctrl[1:-1]:
        num = abs(dy * (p[0] - x0) - dx * (p[1] - y0))
        d = num / denom
        mx = max(mx, d)
    return mx


def aabb_of_points(pts: Sequence[Point]):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def aabb_overlap(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def segment_circle_intersect(a: Point, b: Point, center: Point, r: float) -> bool:
    (x1, y1), (x2, y2) = a, b
    (cx, cy) = center
    dx, dy = x2 - x1, y2 - y1
    if dx == 0.0 and dy == 0.0:
        return math.hypot(cx - x1, cy - y1) <= r + 1e-12
    t = ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px, py = x1 + t * dx, y1 + t * dy
    return math.hypot(px - cx, py - cy) <= r + 1e-12


@dataclass
class Obstacle:
    circle: Tuple[Point, float]  # (center, radius)

    def aabb(self):
        (c, r) = self.circle
        (cx, cy) = c
        return (cx - r, cy - r, cx + r, cy + r)

    def aabb_overlap(self, aabb) -> bool:
        return aabb_overlap(self.aabb(), aabb)

    def seg_hit(self, a: Point, b: Point) -> bool:
        (c, r) = self.circle
        return segment_circle_intersect(a, b, c, r)


def find_intervals_bezier_hits(
    ctrl: Sequence[Point],
    obstacles: List[Obstacle],
    t0: float = 0.0,
    t1: float = 1.0,
    flat_eps: float = 1e-2,
    max_depth: int = 28,
):
    """디카스텔쥬 분할 + AABB 프루닝으로 충돌 의심 t-구간 찾기"""
    hits = []

    def rec(ctrl_local, a, b, depth):
        aabb = aabb_of_points(ctrl_local)
        if not any(obs.aabb_overlap(aabb) for obs in obstacles):
            return
        if (depth >= max_depth) or (bezier_flatness(ctrl_local) <= flat_eps):
            p0, p1 = ctrl_local[0], ctrl_local[-1]
            if any(obs.seg_hit(p0, p1) for obs in obstacles):
                hits.append((a, b))
            return
        L, R = de_casteljau_split(ctrl_local, 0.5)
        m = 0.5 * (a + b)
        rec(L, a, m, depth + 1)
        rec(R, m, b, depth + 1)

    rec(ctrl, t0, t1, 0)
    hits.sort()

    # 구간 병합
    merged = []
    for seg in hits:
        if not merged or seg[0] > merged[-1][1] + 1e-6:
            merged.append(list(seg))
        else:
            merged[-1][1] = max(merged[-1][1], seg[1])
    return [(float(a), float(b)) for (a, b) in merged]


# ------------------------------
# ROS2 노드
# ------------------------------
class PathGenerator(Node):
    """
    클릭된 지점을 기반으로 3차 베지어 곡선을 생성하고,
    /map으로부터 장애물 정보를 구독하여 /path로 발행하는 ROS 2 노드.
    """

    def __init__(self):
        super().__init__("path_generator")

        # === 퍼블리셔 ===
        self.publisher_ = self.create_publisher(Path, "/path", 10)

        # === 구독자 ===
        self.create_subscription(
            PointStamped, "/clicked_point", self.clicked_point_callback, 10
        )

        # QoS: map_server는 보통 Transient Local(Latched) + Reliable
        qos_map = QoSProfile(depth=1)
        qos_map.reliability = ReliabilityPolicy.RELIABLE
        qos_map.durability  = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, "/map", self.map_callback, qos_map)  # ★ QoS 적용

        # === 파라미터(필요하면 ros2 param으로 바꿀 수 있음) ===
        self.robot_radius = 0.20  # [m] 로봇 반경
        self.safety_margin = 0.05  # [m] 추가 여유
        self.samples_per_curve = 80  # Path 샘플 개수
        self.flat_eps = 5e-3  # 디카스텔쥬 평탄성 임계
        self.occ_threshold = 100  # 점유판정 기준(보통 100: 확실한 점유)

        # === 내부 상태 ===
        self.cp_list: List[np.ndarray] = []  # 클릭된 제어점 리스트(3D)
        self.map_info = None  # 지도 메타정보 (해상도, origin 등)
        self.obstacles: List[Obstacle] = []  # 원형 장애물 리스트

        self.get_logger().info(
            "✅ PathGenerator 노드 실행: /clicked_point + /map 구독, /path 발행"
        )

    # ----------------------------------------------------------------
    # /map 콜백: OccupancyGrid를 받아 장애물 정보를 원(인플레이트)로 추출
    # ----------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        self.map_info = msg.info
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int16).reshape(height, width)

        # 셀 대각선의 절반 + 로봇 반경 + 마진 => 보수적 반경
        r_cell = res * math.sqrt(2) * 0.5
        inflated_r = r_cell + self.robot_radius + self.safety_margin

        obstacles: List[Obstacle] = []
        # OccupancyGrid 데이터는 (row=i(y), col=j(x))로 origin에서 +x, +y로 증가
        occ_idx = np.argwhere(grid >= self.occ_threshold)
        for (iy, ix) in occ_idx:
            cx = ox + (ix + 0.5) * res
            cy = oy + (iy + 0.5) * res
            obstacles.append(Obstacle(circle=((cx, cy), inflated_r)))

        self.obstacles = obstacles
        self.get_logger().info(
            f"/map 갱신 — 장애물(원) {len(obstacles)}개, r≈{inflated_r:.3f} m"
        )

    # ----------------------------------------------------------------
    # /clicked_point 콜백: 4개 모이면 3차 베지어 1세그먼트 생성
    # ----------------------------------------------------------------
    def clicked_point_callback(self, msg: PointStamped):
        self.cp_list.append(np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float))
        self.get_logger().info(
            f"[{len(self.cp_list)}] 지점 추가: x={msg.point.x:.3f}, y={msg.point.y:.3f}"
        )

        if len(self.cp_list) % 4 == 0:
            if self.map_info is None:
                self.get_logger().warn("⚠️ 아직 /map 데이터를 받지 못함. 맵 수신 후 생성하세요.")
                return

            last4_cps = self.cp_list[-4:]
            self.get_logger().info("🟦 4개 지점 수신 → 안전 베지어 곡선 생성/발행")
            self.generate_and_publish_bezier_path(last4_cps)

    # ----------------------------------------------------------------
    # 안전 곡선 계획기: 충돌구간 찾고 내부 CP(1,2) 외측으로 살짝 밀기(간단 근사)
    # ----------------------------------------------------------------
    def plan_safe_curve(self, ctrl_3d: Sequence[np.ndarray]) -> List[Point]:
        """
        입력: 4개 제어점(3D), 출력: 충돌 완화 후 2D(x,y) 제어점
        - 장애물 없으면 그대로 반환
        - 간단 반복으로 충돌 구간의 중점/양끝에서 외측법선 방향으로 내부 CP 두 개를 이동
        """
        if len(ctrl_3d) != 4:
            return [(float(p[0]), float(p[1])) for p in ctrl_3d]

        ctrl2d: List[Point] = [(float(p[0]), float(p[1])) for p in ctrl_3d]
        obstacles = self.obstacles
        if not obstacles:
            return ctrl2d

        max_passes = 40          # 반복 강화
        base_step = 0.05         # 최소 밀어내기
        k = 0.8                  # 침투량 비례 계수

        for _ in range(max_passes):
            intervals = find_intervals_bezier_hits(
                ctrl2d, obstacles, flat_eps=self.flat_eps, max_depth=28
            )
            if not intervals:
                break  # 충돌 없음

            moved = False
            for (a, b) in intervals:
                # 구간의 시작/중간/끝 3지점을 모두 보정
                for tmid in (a, 0.5 * (a + b), b):
                    xmid, ymid = bezier_eval(ctrl2d, tmid)

                    # 중점에서 가장 가까운 원형 장애물 찾기
                    best_obs = None
                    best_d = 1e9
                    best_c = None
                    best_r = None
                    for obs in obstacles:
                        (c, r) = obs.circle
                        d = math.hypot(xmid - c[0], ymid - c[1]) - r
                        if d < best_d:
                            best_d = d
                            best_obs = obs
                            best_c, best_r = c, r
                    if best_obs is None:
                        continue

                    # 외측 법선
                    vx, vy = (xmid - best_c[0], ymid - best_c[1])
                    nrm = math.hypot(vx, vy) + 1e-12
                    nx, ny = vx / nrm, vy / nrm

                    # 침투량 기반 가변 push
                    dist = math.hypot(xmid - best_c[0], ymid - best_c[1])
                    penetr = max(0.0, best_r - dist)  # 장애물 내부에 있으면 양수
                    push = max(base_step, k * (penetr + self.safety_margin))

                    # 내부 제어점(1,2)만 이동
                    P = list(ctrl2d)
                    P[1] = (P[1][0] + push * nx, P[1][1] + push * ny)
                    P[2] = (P[2][0] + push * nx, P[2][1] + push * ny)
                    ctrl2d = P
                    moved = True

            # 이번 패스에서 전혀 못 밀었으면 step을 키워 탈출 유도
            if not moved:
                base_step *= 1.5

        return ctrl2d

    # ----------------------------------------------------------------
    # 베지어 곡선 생성(안전 보정 포함) 및 Path 발행
    # ----------------------------------------------------------------
    def generate_and_publish_bezier_path(self, control_points: Sequence[np.ndarray]):
        # 1) 2D 안전 곡선으로 CP 보정
        safe_ctrl2d = self.plan_safe_curve(control_points)

        # 2) Path 메시지 구성
        path_msg = Path()
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        path_msg.header.frame_id = "map"

        poses: List[PoseStamped] = []
        N = max(10, int(self.samples_per_curve))
        for i in range(N):
            t = float(i) / float(N - 1)
            x, y = bezier_eval(safe_ctrl2d, t)
            # z는 입력 CP z를 그대로 베지어로 보간
            z = float(
                (1 - t) ** 3 * control_points[0][2]
                + 3 * (1 - t) ** 2 * t * control_points[1][2]
                + 3 * (1 - t) * t ** 2 * control_points[2][2]
                + t ** 3 * control_points[3][2]
            )

            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "map"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = z
            ps.pose.orientation.w = 1.0
            poses.append(ps)

        path_msg.poses = poses
        self.publisher_.publish(path_msg)

        # 남은 충돌 구간이 있는지 로그로 확인(보수적)
        remain = find_intervals_bezier_hits(
            safe_ctrl2d, self.obstacles, flat_eps=self.flat_eps, max_depth=28
        )
        if remain:
            self.get_logger().warn(
                f"⚠️ 충돌 의심 구간 잔존: {[(round(a,3), round(b,3)) for (a,b) in remain]}"
            )
        else:
            self.get_logger().info(f"✅ 안전 경로 발행 완료: 샘플 {len(poses)}개")

    # ----------------------------------------------------------------
    # (선택) 타이머 기반 주기 재계획을 원하면 create_timer로 콜백 추가 가능
    # ----------------------------------------------------------------
    # def periodic_replan(self):
    #     if len(self.cp_list) >= 4 and self.map_info is not None:
    #         self.generate_and_publish_bezier_path(self.cp_list[-4:])


# ------------------------------
# main
# ------------------------------
def main(args=None):
    rp.init(args=args)
    node = PathGenerator()
    try:
        rp.spin(node)
    finally:
        node.destroy_node()
        rp.shutdown()


if __name__ == "__main__":
    main()
