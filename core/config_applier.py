import logging

from core.config_manager import diff_configs

logger = logging.getLogger(__name__)


def apply_config_changes(orchestrator, space_logger, new_config):
    """설정 변경 감지 → 구조적 변경 적용 + 값 변경 시 카메라 재시작."""
    old = orchestrator.app_config
    diff = diff_configs(old, new_config)

    # config 먼저 적용 → downstream 동작이 최신 설정 기준으로 동작
    orchestrator.update_config(new_config)

    # 1. camera/space 구조 변경
    for cam_id, (old_space, new_space) in diff.reassigned_cameras.items():
        logger.info("Camera %s reassigned: %s → %s", cam_id, old_space, new_space)
        orchestrator.reassign_camera(cam_id, old_space, new_space)
        if orchestrator._vision_detector:
            if old_space:
                orchestrator._vision_detector.remove_camera_from_space(old_space, cam_id)
            if new_space:
                orchestrator._vision_detector.add_camera_to_space(new_space, cam_id)

    for cam_id in diff.added_cameras:
        cam = next((c for c in new_config.cameras if c.id == cam_id), None)
        if cam:
            orchestrator.add_camera(cam)

    for cam_id in diff.removed_cameras:
        orchestrator.remove_camera(cam_id)

    for space_id in diff.added_spaces:
        space = next((s for s in new_config.spaces if s.id == space_id), None)
        if space:
            logger.info("Space added: %s (cameras: %d)", space_id, len(space.camera_ids))
            if orchestrator._vision_detector:
                orchestrator._vision_detector.add_space(space)

    for space_id in diff.removed_spaces:
        if orchestrator._vision_detector:
            orchestrator._vision_detector.remove_space(space_id)

    # 2. 설정 값 변경 감지 → 영향받는 카메라 재시작
    needs_restart = False

    if old.llm != new_config.llm:
        logger.info("LLM config changed — restarting cameras")
        needs_restart = True

    if old.reconnect != new_config.reconnect:
        logger.info("Reconnect config changed — restarting cameras")
        needs_restart = True

    # camera 개별 설정 변경 감지
    for cam in new_config.cameras:
        old_cam = next((c for c in old.cameras if c.id == cam.id), None)
        if old_cam and camera_values_differ(old_cam, cam):
            logger.info("Camera %s settings changed — restarting", cam.id)
            needs_restart = True

    if needs_restart:
        _restart_all_cameras(orchestrator, space_logger, new_config)


def camera_values_differ(old_cam, new_cam):
    """두 CameraConfig의 설정 값이 다른지 비교."""
    for attr in ("source", "status", "target_classes", "llm_system_prompt"):
        if getattr(old_cam, attr) != getattr(new_cam, attr):
            return True
    return False


def _restart_all_cameras(orchestrator, space_logger, new_config):
    """모든 카메라 중지 → config 업데이트 → 재시작."""
    cam_ids = list(orchestrator._collectors.keys())

    for cam_id in cam_ids:
        orchestrator.remove_camera(cam_id)

    orchestrator.update_config(new_config)

    for cam in new_config.cameras:
        if cam.status == "active":
            orchestrator.add_camera(cam)
