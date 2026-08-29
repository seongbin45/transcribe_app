"""main_window.py와 settings_dialog.py가 함께 쓰는 상수 (순환 import 방지용)."""

MODEL_CHOICES = [
    ("small (빠른 테스트용)", "small"),
    ("medium (균형)", "medium"),
    ("large-v3 (권장, 느림·최초 실행 시 다운로드)", "large-v3"),
]

EXPORT_FORMATS = [
    ("Word (.docx)", "docx", "Word 문서 (*.docx)"),
    ("Markdown (.md)", "md", "Markdown 파일 (*.md)"),
    ("텍스트 (.txt)", "txt", "텍스트 파일 (*.txt)"),
    ("자막 (.srt)", "srt", "SRT 자막 (*.srt)"),
    ("자막 (.vtt)", "vtt", "WebVTT 자막 (*.vtt)"),
]
