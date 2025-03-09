import numpy as np
import psutil
import time
import torch
import typer

from pathlib import Path
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.panel import Panel
from rich.table import Table
from typing import List, Tuple
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from utils.factory import ModelFactory, ConfigCreator, ModelConfig
from utils.inference import (
    Torchvision_Inference,
    ImageProcessor,
    InferenceConfig,
    PatchConfig,
    ProcessorConfig
)

# Initialize Typer app
app = typer.Typer(help="CLI for running object detection inference on medical images using trained models.")
console = Console()

def load_model_from_config(config_path: Path):
    """Load model from configuration file."""
    try:
        config = ConfigCreator.load(str(config_path))
        model = ModelFactory.load(config)
        return model, config
    except Exception as e:
        console.print(f"[red]Error loading model: {str(e)}[/red]")
        raise typer.Exit(1)

def setup_inference(
    model: torch.nn.Module,
    is_wsi: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    patch_size: int,
    overlap: float,
    overwrite: bool
) -> tuple:
    """Setup inference components."""
    inference_config = InferenceConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        is_wsi=is_wsi
    )

    patch_config = PatchConfig(
        size=patch_size,
        overlap=overlap
    )

    processor_config = ProcessorConfig(
        overwrite=overwrite
    )

    strategy = Torchvision_Inference(model, inference_config)
    processor = ImageProcessor(
        strategy,
        processor_config
    )

    return processor, patch_config


def print_model_config(config: ModelConfig) -> None:
    """Print model configuration in a nicely formatted table."""
    # Create main configuration table
    main_table = Table(
        title="[bold]Model Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    main_table.add_column("Parameter", style="bold green")
    main_table.add_column("Value", style="yellow")

    # Add basic configuration rows
    basic_params = [
        ("Model Name", config.model_name),
        ("Detector Type", config.detector),
        ("Backbone", config.backbone),
        ("Number of Classes", config.num_classes),
        ("Detection Threshold", f"{config.det_thresh:.3f}"),
        ("Extra Blocks", str(config.extra_blocks)),
        ("Patch Size", str(config.patch_size)),
    ]

    for param, value in basic_params:
        main_table.add_row(param, str(value))

    # Add optional parameters if they exist
    if hasattr(config, 'weights') and config.weights:
        main_table.add_row("Initial Weights", str(config.weights))

    if hasattr(config, 'returned_layers') and config.returned_layers:
        main_table.add_row("Returned Layers", str(config.returned_layers))

    # Special handling for detector-specific parameters
    if hasattr(config, 'anchor_sizes') and config.anchor_sizes:
        main_table.add_row("Anchor Sizes", str(config.anchor_sizes))

    if hasattr(config, 'anchor_ratios') and config.anchor_ratios:
        main_table.add_row("Anchor Ratios", str(config.anchor_ratios))

    # Print the configuration
    console.print("\n")
    console.print(Panel(main_table, title="[bold blue]Model Configuration",
                       subtitle="[bold blue]Detection Pipeline"))
    console.print("\n")



class PerformanceMonitor:
    def __init__(self):
        self.inference_times: List[float] = []
        self.memory_usage: List[Tuple[float, float]] = []  # (GPU memory, CPU memory)
        self.console = Console()

    def get_gpu_memory(self) -> float:
        """Get GPU memory usage in GB."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            return memory_allocated, memory_reserved
        return 0.0, 0.0

    def get_cpu_memory(self) -> float:
        """Get CPU memory usage in GB."""
        return psutil.Process().memory_info().rss / 1024**3

    def record_iteration(self, inference_time: float):
        """Record performance metrics for one iteration."""
        self.inference_times.append(inference_time)
        gpu_allocated, gpu_reserved = self.get_gpu_memory()
        cpu_usage = self.get_cpu_memory()
        self.memory_usage.append((gpu_allocated, cpu_usage))

    def print_summary(self):
        """Print a summary of performance metrics."""
        # Create performance table
        table = Table(title="Performance Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Calculate statistics
        avg_time = np.mean(self.inference_times)
        std_time = np.std(self.inference_times)
        max_gpu_allocated = max(m[0] for m in self.memory_usage)
        avg_gpu_allocated = np.mean([m[0] for m in self.memory_usage])
        max_cpu_usage = max(m[1] for m in self.memory_usage)
        avg_cpu_usage = np.mean([m[1] for m in self.memory_usage])

        # Add rows to table
        table.add_row("Average Time per Image", f"{avg_time:.3f} seconds")
        table.add_row("Std Dev Time", f"{std_time:.3f} seconds")
        table.add_row("Max GPU Memory Allocated", f"{max_gpu_allocated:.2f} GB")
        table.add_row("Average GPU Memory", f"{avg_gpu_allocated:.2f} GB")
        table.add_row("Max CPU Memory", f"{max_cpu_usage:.2f} GB")
        table.add_row("Average CPU Memory", f"{avg_cpu_usage:.2f} GB")
        table.add_row("Total Images Processed", str(len(self.inference_times)))

        self.console.print("\n")
        self.console.print(table)



@app.command()
def detect(
    config_path: Path = typer.Argument(
        ...,
        help="Path to model configuration YAML file",
        exists=True
    ),
    input_path: Path = typer.Argument(
        ...,
        help="Path to input image or directory"
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory to save detection results"
    ),
    is_wsi: bool = typer.Option(
        False,
        "--wsi",
        help="Process whole slide images"
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        help="Batch size for inference"
    ),
    num_workers: int = typer.Option(
        4,
        "--workers",
        "-w",
        help="Number of worker processes"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        "-d",
        help="Device for inference"
    ),
    patch_size: int = typer.Option(
        1024,
        "--patch-size",
        "-p",
        help="Size of image patches"
    ),
    overlap: float = typer.Option(
        0.3,
        "--overlap",
        "-ol",
        help="Overlap between patches"
    ),
    overwrite: bool = typer.Option(
        False, 
        "--overwrite",
        "-ow",
        help="If existing results should be overwritten."
    ),
    measure_performance: bool = typer.Option(
        False, 
        "--measure-performance", 
        "-m", 
        help="Measure and report performance metrics"
    )
) -> None:
    """
    Perform object detection on images using a trained model.
    """
    try:
        # Load model
        with console.status("[bold green]Loading model...") as status:
            model, config = load_model_from_config(config_path)
            status.update("[bold green]Model loaded successfully!")

        # Print configuration
        print_model_config(config)

        # Print runtime settings
        runtime_table = Table(
            title="[bold]Runtime Settings",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        runtime_table.add_column("Parameter", style="bold green")
        runtime_table.add_column("Value", style="yellow")

        runtime_params = [
            ("Processing Mode", "Whole Slide Image" if is_wsi else "Regular Image"),
            ("Batch Size", str(batch_size)),
            ("Number of Workers", str(num_workers)),
            ("Device", device),
            ("Patch Size", str(patch_size)),
            ("Patch Overlap", f"{overlap:.2f}"),
            ("Input Path", str(input_path)),
            ("Output Directory", str(output_dir)),
            ("Measure Performance", "Yes" if measure_performance else "No")
        ]

        for param, value in runtime_params:
            runtime_table.add_row(param, value)

        console.print(Panel(runtime_table, title="[bold blue]Runtime Configuration"))
        console.print("\n")

        # Setup inference components
        processor, patch_config = setup_inference(
            model, is_wsi, batch_size, num_workers,
            device, patch_size, overlap, overwrite
        )

        # Define valid extensions based on image type
        wsi_extensions = ('.svs', '.tif', '.tiff', '.dcm', '.vms', '.ndpi', '.vmu', '.mrxs', '.czi')
        regular_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
        valid_extensions = wsi_extensions if is_wsi else regular_extensions

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize performance monitor if requested
        perf_monitor = PerformanceMonitor() if measure_performance else None

        # Process images
        if input_path.is_file():
            # Check if file extension is valid
            if input_path.suffix.lower() not in valid_extensions:
                console.print(f"[red]Invalid file type! Expected one of: {', '.join(valid_extensions)}[/red]")
                raise typer.Exit(1)

            # Single image processing
            console.print(f"\n[bold]Processing single image: {input_path.name}[/bold]")

            if measure_performance:
                start_time = time.time()
                processor.process_single(input_path, patch_config=patch_config, output_dir=output_dir)
                inference_time = time.time() - start_time
                perf_monitor.record_iteration(inference_time)
            else:
                processor.process_single(input_path, patch_config=patch_config, output_dir=output_dir)
            
            console.print(f"[green]Results saved to: {output_dir / f'{input_path.stem}_detections.json'}[/green]")
            

        elif input_path.is_dir():
            # Batch processing
            image_paths = []
            for ext in valid_extensions:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))  # Handle uppercase extensions

            if not image_paths:
                console.print(f"[red]No {'WSI' if is_wsi else 'regular'} images found in directory![/red]")
                console.print(f"[red]Supported extensions: {', '.join(valid_extensions)}[/red]")
                raise typer.Exit(1)

            console.print(f"\n[bold]Found {len(image_paths)} valid images[/bold]")
            console.print(f"[bold]Processing {len(image_paths)} images...[/bold]")
            
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Processing {len(image_paths)} images...", total=len(image_paths))

                for img_path in sorted(image_paths):
                    if measure_performance:
                        start_time = time.time()
                        processor.process_single(img_path, patch_config=patch_config, output_dir=output_dir)
                        inference_time = time.time() - start_time
                        perf_monitor.record_iteration(inference_time)
                    else:
                        processor.process_single(img_path, patch_config=patch_config, output_dir=output_dir)
                    progress.advance(task)

            console.print(f"[green]Results saved to: {output_dir}[/green]")

        else:
            console.print("[red]Invalid input path![/red]")
            raise typer.Exit(1)
        
        # Print performance summary if requested
        if measure_performance:
            perf_monitor.print_summary()

    except Exception as e:
        console.print(f"[red]Error during processing: {str(e)}[/red]")
        raise typer.Exit(1)
    
@app.command()
def validate_config(
    config_path: Path = typer.Argument(
        ...,
        help="Path to model configuration YAML file",
        exists=True
    )
) -> None:
    """
    Validate a model configuration file.
    """
    try:
        with console.status("[bold green]Validating configuration...") as status:
            config = ConfigCreator.load(str(config_path))
            status.update("[bold green]Configuration loaded successfully!")

        print_model_config(config)
        console.print("[green]✓ Configuration is valid![/green]")

    except Exception as e:
        console.print(f"[red]Invalid configuration: {str(e)}[/red]")
        raise typer.Exit(1)
 
if __name__ == "__main__":
    app()