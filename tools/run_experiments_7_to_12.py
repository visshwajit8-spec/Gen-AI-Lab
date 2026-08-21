"""
Runner script for Experiments 7 to 12.
Executes each experiment, generates outputs and terminal screenshots.
"""
import subprocess
import sys
from pathlib import Path
from tools.make_screenshot import render_screenshot

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    {
        "id": "exp7",
        "folder": ROOT / "Experiment-07-Code-Generation-Debugging",
        "script": ROOT / "Experiment-07-Code-Generation-Debugging" / "source" / "exp7_code_generation.py",
        "output": ROOT / "Experiment-07-Code-Generation-Debugging" / "output" / "code_generation_results.txt",
        "screenshot": ROOT / "Experiment-07-Code-Generation-Debugging" / "screenshots" / "code_generation_output.png",
        "title": "CS4V48 - Ex. 7: AI-Powered Code Generation & Debugging",
    },
    {
        "id": "exp8",
        "folder": ROOT / "Experiment-08-Image-Generation-Diffusion",
        "script": ROOT / "Experiment-08-Image-Generation-Diffusion" / "source" / "exp8_image_diffusion.py",
        "output": ROOT / "Experiment-08-Image-Generation-Diffusion" / "output" / "image_generation_results.txt",
        "screenshot": ROOT / "Experiment-08-Image-Generation-Diffusion" / "screenshots" / "image_diffusion_output.png",
        "title": "CS4V48 - Ex. 8: Image Generation Using Diffusion Models",
    },
    {
        "id": "exp9",
        "folder": ROOT / "Experiment-09-Multimodal-Text-Image",
        "script": ROOT / "Experiment-09-Multimodal-Text-Image" / "source" / "exp9_multimodal_blip.py",
        "output": ROOT / "Experiment-09-Multimodal-Text-Image" / "output" / "multimodal_results.txt",
        "screenshot": ROOT / "Experiment-09-Multimodal-Text-Image" / "screenshots" / "multimodal_output.png",
        "title": "CS4V48 - Ex. 9: Multimodal AI (BLIP Captioning & VQA)",
    },
    {
        "id": "exp10",
        "folder": ROOT / "Experiment-10-Fine-Tuning-Domain-Specific",
        "script": ROOT / "Experiment-10-Fine-Tuning-Domain-Specific" / "source" / "exp10_fine_tuning.py",
        "output": ROOT / "Experiment-10-Fine-Tuning-Domain-Specific" / "output" / "fine_tuning_results.txt",
        "screenshot": ROOT / "Experiment-10-Fine-Tuning-Domain-Specific" / "screenshots" / "fine_tuning_output.png",
        "title": "CS4V48 - Ex. 10: Fine-Tuning Pre-Trained Language Model",
    },
    {
        "id": "exp11",
        "folder": ROOT / "Experiment-11-Content-Generation-Multimedia",
        "script": ROOT / "Experiment-11-Content-Generation-Multimedia" / "source" / "exp11_multimedia_content_generation.py",
        "output": ROOT / "Experiment-11-Content-Generation-Multimedia" / "output" / "content_generation_results.txt",
        "screenshot": ROOT / "Experiment-11-Content-Generation-Multimedia" / "screenshots" / "content_generation_output.png",
        "title": "CS4V48 - Ex. 11: AI Multimedia Content Generation System",
    },
    {
        "id": "exp12",
        "folder": ROOT / "Experiment-12-Deployment-And-Evaluation",
        "script": ROOT / "Experiment-12-Deployment-And-Evaluation" / "source" / "exp12_deployment_evaluation.py",
        "output": ROOT / "Experiment-12-Deployment-And-Evaluation" / "output" / "deployment_evaluation_results.txt",
        "screenshot": ROOT / "Experiment-12-Deployment-And-Evaluation" / "screenshots" / "deployment_evaluation_output.png",
        "title": "CS4V48 - Ex. 12: Deployment & ROUGE Evaluation",
    },
]


def run_all():
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running {exp['title']}...")
        print(f"{'='*60}")
        result = subprocess.run([sys.executable, str(exp["script"])], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if exp["output"].exists():
            render_screenshot(str(exp["output"]), str(exp["screenshot"]), exp["title"])
            print(f"Generated screenshot: {exp['screenshot']}")
        else:
            print(f"Warning: Output file {exp['output']} was not created.")


if __name__ == "__main__":
    run_all()
