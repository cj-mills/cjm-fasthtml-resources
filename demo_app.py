"""Comprehensive demo application for cjm-fasthtml-resources library.

This demo showcases all library features:
- ResourceManager for tracking worker processes
- Resource validation for job execution
- GPU and CPU conflict detection
- Plugin switching and resource management
- Configuration schemas
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from fasthtml.common import *
from cjm_fasthtml_daisyui.core.resources import get_daisyui_headers
from cjm_fasthtml_daisyui.core.testing import create_theme_persistence_script

print("\n" + "="*70)
print("Initializing cjm-fasthtml-resources Demo")
print("="*70)

# Step 1: Import library components
print("\n[1/5] Importing library components...")
from cjm_fasthtml_resources.core.manager import (
    ResourceManager,
    ResourceType,
    ResourceStatus,
    ResourceConflict,
    WorkerState
)
from cjm_fasthtml_resources.core.validation import (
    ValidationAction,
    ValidationResult,
    validate_resources_for_job
)
from cjm_fasthtml_resources.core.monitoring_config import RESOURCE_MONITOR_SCHEMA
from cjm_fasthtml_resources.core.management_config import RESOURCE_MANAGEMENT_SCHEMA
from cjm_fasthtml_resources.utils.plugin_utils import (
    is_local_model_plugin,
    uses_gpu_device,
    get_plugin_resource_identifier,
    compare_plugin_resources,
    get_plugin_resource_requirements
)
print("  ✓ All library components imported successfully")


# Step 2: Create mock plugin registry for demo
print("\n[2/5] Setting up mock plugin registry...")

@dataclass
class MockPluginMetadata:
    """Mock plugin metadata."""
    name: str
    config_schema: Dict[str, Any]

    def get_unique_id(self) -> str:
        return f"mock-{self.name}"


class MockPluginRegistry:
    """Mock plugin registry for demo."""

    def __init__(self):
        self.plugins = [
            MockPluginMetadata(
                name="whisper-local",
                config_schema={
                    "properties": {
                        "model_id": {"type": "string"},
                        "device": {"type": "string"}
                    }
                }
            ),
            MockPluginMetadata(
                name="whisper-api",
                config_schema={
                    "properties": {
                        "api_key": {"type": "string"},
                        "model": {"type": "string"}
                    }
                }
            ),
            MockPluginMetadata(
                name="llama-local",
                config_schema={
                    "properties": {
                        "model_id": {"type": "string"},
                        "device": {"type": "string"}
                    }
                }
            ),
        ]

        self.configs = {
            "mock-whisper-local": {"model_id": "whisper-large-v3", "device": "cuda"},
            "mock-whisper-api": {"api_key": "sk-xxx", "model": "whisper-1"},
            "mock-llama-local": {"model_id": "llama-3.1-8b", "device": "cuda"},
        }

    def get_plugin(self, plugin_id: str) -> Optional[MockPluginMetadata]:
        """Get plugin by ID."""
        for plugin in self.plugins:
            if plugin.get_unique_id() == plugin_id:
                return plugin
        return None

    def load_plugin_config(self, plugin_id: str) -> Dict[str, Any]:
        """Load plugin configuration."""
        return self.configs.get(plugin_id, {})


plugin_registry = MockPluginRegistry()
print("  ✓ Mock plugin registry created with 3 plugins")


# Step 3: Create resource manager
print("\n[3/5] Creating resource manager...")
resource_manager = ResourceManager(gpu_memory_threshold_percent=45.0)
print("  ✓ Resource manager initialized")
print(f"    GPU memory threshold: 45%")


# Step 4: Create helper functions for validation
print("\n[4/5] Setting up validation helpers...")

def get_requirements_helper(plugin_id: str, plugin_config: Optional[Dict[str, Any]] = None):
    """Helper function for get_plugin_resource_requirements."""
    return get_plugin_resource_requirements(plugin_id, plugin_registry, plugin_config)

print("  ✓ Validation helpers configured")


# Step 5: Set up FastHTML app
print("\n[5/5] Setting up FastHTML application...")


def main():
    """Main entry point - creates the demo app."""

    # Create the FastHTML app
    app, rt = fast_app(
        pico=False,
        hdrs=[
            *get_daisyui_headers(),
            create_theme_persistence_script(),
        ],
        title="FastHTML Resources Demo",
        htmlkw={'data-theme': 'light'}
    )

    from cjm_fasthtml_tailwind.utilities.spacing import p, m
    from cjm_fasthtml_tailwind.utilities.sizing import container, max_w, w
    from cjm_fasthtml_tailwind.utilities.typography import font_size, font_weight, text_align
    from cjm_fasthtml_tailwind.core.base import combine_classes
    from cjm_fasthtml_daisyui.components.actions.button import btn, btn_colors, btn_sizes, btn_styles
    from cjm_fasthtml_daisyui.components.data_display.badge import badge, badge_colors, badge_styles
    from cjm_fasthtml_daisyui.components.data_input.select import select, select_sizes

    @rt("/")
    def get():
        """Homepage with feature showcase."""
        return Main(
            Div(
                H1("cjm-fasthtml-resources Demo",
                   cls=combine_classes(font_size._4xl, font_weight.bold, m.b(4))),

                P("Resource management system for tracking workers and detecting conflicts:",
                  cls=combine_classes(font_size.lg, m.b(6))),

                # Feature list
                Div(
                    Div(
                        Span("✓", cls=combine_classes(font_size._2xl, m.r(3))),
                        Span("ResourceManager for tracking worker processes"),
                        cls=combine_classes(m.b(3))
                    ),
                    Div(
                        Span("✓", cls=combine_classes(font_size._2xl, m.r(3))),
                        Span("GPU and CPU resource conflict detection"),
                        cls=combine_classes(m.b(3))
                    ),
                    Div(
                        Span("✓", cls=combine_classes(font_size._2xl, m.r(3))),
                        Span("Resource validation for job execution"),
                        cls=combine_classes(m.b(3))
                    ),
                    Div(
                        Span("✓", cls=combine_classes(font_size._2xl, m.r(3))),
                        Span("Plugin switching and resource management"),
                        cls=combine_classes(m.b(3))
                    ),
                    Div(
                        Span("✓", cls=combine_classes(font_size._2xl, m.r(3))),
                        Span("Configuration schemas for monitoring and management"),
                        cls=combine_classes(m.b(8))
                    ),
                    cls=combine_classes(text_align.left, m.b(8))
                ),

                # Statistics
                Div(
                    Span(
                        Span(f"{len(plugin_registry.plugins)}", cls=str(font_weight.bold)),
                        " Plugins Available",
                        cls=combine_classes(badge, badge_colors.info, m.r(2))
                    ),
                    Span(
                        Span(f"{len(resource_manager.get_all_workers())}", cls=str(font_weight.bold)),
                        " Active Workers",
                        cls=combine_classes(badge, badge_colors.success, m.r(2))
                    ),
                    Span(
                        Span(f"{len(resource_manager.get_active_worker_types())}", cls=str(font_weight.bold)),
                        " Worker Types",
                        cls=combine_classes(badge, badge_colors.warning)
                    ),
                    cls=combine_classes(m.b(8))
                ),

                A(
                    "Try the Resource Manager",
                    href="/manager",
                    cls=combine_classes(btn, btn_colors.primary, btn_sizes.lg)
                ),

                cls=combine_classes(
                    container,
                    max_w._4xl,
                    m.x.auto,
                    p(8),
                    text_align.center
                )
            )
        )

    @rt("/manager")
    def get():
        """Interactive resource manager page."""
        return Main(
            Div(
                H1("Resource Manager Demo",
                   cls=combine_classes(font_size._3xl, font_weight.bold, m.b(6))),

                # Worker registration form
                Div(
                    H2("Register Worker", cls=combine_classes(font_size._2xl, font_weight.bold, m.b(4))),
                    Form(
                        Div(
                            Div(
                                Label("Worker Type:", cls=combine_classes(font_weight.bold, m.b(2))),
                                Select(
                                    Option("Transcription", value="transcription"),
                                    Option("LLM", value="llm"),
                                    Option("Image Generation", value="image_gen"),
                                    name="worker_type",
                                    cls=combine_classes(select, select_sizes.md, w.full, m.b(4))
                                ),
                                cls=combine_classes(m.b(4))
                            ),
                            Div(
                                Label("Plugin:", cls=combine_classes(font_weight.bold, m.b(2))),
                                Select(
                                    Option("Whisper (Local GPU)", value="mock-whisper-local"),
                                    Option("Whisper (API)", value="mock-whisper-api"),
                                    Option("LLaMA (Local GPU)", value="mock-llama-local"),
                                    name="plugin_id",
                                    cls=combine_classes(select, select_sizes.md, w.full, m.b(4))
                                ),
                                cls=combine_classes(m.b(4))
                            ),
                        ),
                        Button(
                            "Register Worker",
                            type="submit",
                            cls=combine_classes(btn, btn_colors.primary, btn_sizes.md)
                        ),
                        hx_post="/workers/register",
                        hx_target="#workers-list",
                        hx_swap="outerHTML"
                    ),
                    cls=combine_classes(m.b(8), p(6), "bg-base-200", "rounded-lg")
                ),

                # Workers list
                Div(
                    id="workers-list",
                    hx_get="/workers/list",
                    hx_trigger="load",
                    hx_swap="outerHTML",
                    cls=combine_classes(m.b(4))
                ),

                # Validation tester
                Div(
                    H2("Test Resource Validation", cls=combine_classes(font_size._2xl, font_weight.bold, m.b(4))),
                    Form(
                        Div(
                            Label("Plugin to validate:", cls=combine_classes(font_weight.bold, m.b(2))),
                            Select(
                                Option("Whisper (Local GPU)", value="mock-whisper-local"),
                                Option("Whisper (API)", value="mock-whisper-api"),
                                Option("LLaMA (Local GPU)", value="mock-llama-local"),
                                name="plugin_id",
                                cls=combine_classes(select, select_sizes.md, w.full, m.b(4))
                            ),
                        ),
                        Button(
                            "Validate Resources",
                            type="submit",
                            cls=combine_classes(btn, btn_colors.secondary, btn_sizes.md)
                        ),
                        hx_post="/validate",
                        hx_target="#validation-result",
                        hx_swap="outerHTML"
                    ),
                    Div(id="validation-result", cls=combine_classes(m.t(4))),
                    cls=combine_classes(m.b(8), p(6), "bg-base-200", "rounded-lg")
                ),

                # Back to home
                A(
                    "← Back to Home",
                    href="/",
                    cls=combine_classes(btn, btn_styles.ghost, m.t(4))
                ),

                cls=combine_classes(
                    container,
                    max_w._4xl,
                    m.x.auto,
                    p(8)
                )
            )
        )

    @rt("/workers/register")
    def post(worker_type: str, plugin_id: str):
        """Register a new worker."""
        import random

        # Generate fake PID
        pid = random.randint(10000, 99999)

        # Get plugin info
        plugin = plugin_registry.get_plugin(plugin_id)
        config = plugin_registry.load_plugin_config(plugin_id)

        # Extract resource identifier
        resource_id = get_plugin_resource_identifier(config)

        # Register worker
        resource_manager.register_worker(
            pid=pid,
            worker_type=worker_type,
            plugin_id=plugin_id,
            plugin_name=plugin.name if plugin else "unknown",
            loaded_plugin_resource=resource_id,
            config=config
        )

        print(f"[Demo] Registered {worker_type} worker PID {pid} with plugin {plugin.name if plugin else 'unknown'}")

        return workers_list()

    @rt("/workers/list")
    def workers_list():
        """List all registered workers."""
        from cjm_fasthtml_daisyui.components.data_display.table import table, table_modifiers

        workers = resource_manager.get_all_workers()

        if not workers:
            return Div(
                H2("Active Workers", cls=combine_classes(font_size._2xl, font_weight.bold, m.b(4))),
                P("No workers registered yet. Register one above!",
                  cls=combine_classes("text-gray-500", "italic")),
                id="workers-list"
            )

        # Create table rows
        rows = []
        for worker in workers:
            type_badge = badge_colors.info
            if worker.worker_type == "transcription":
                type_badge = badge_colors.primary
            elif worker.worker_type == "llm":
                type_badge = badge_colors.secondary

            status_badge = badge_colors.success if worker.status == "idle" else badge_colors.warning

            # Check if uses GPU
            uses_gpu = uses_gpu_device(worker.config) if worker.config else False
            device_text = worker.config.get('device', 'N/A') if worker.config else 'N/A'

            rows.append(
                Tr(
                    Td(str(worker.pid)),
                    Td(Span(worker.worker_type, cls=combine_classes(badge, type_badge))),
                    Td(worker.plugin_name or "N/A"),
                    Td(worker.loaded_plugin_resource or "N/A"),
                    Td(device_text),
                    Td(Span(worker.status, cls=combine_classes(badge, status_badge))),
                    Td(
                        Button(
                            "Remove",
                            hx_post=f"/workers/{worker.pid}/remove",
                            hx_target="#workers-list",
                            hx_swap="outerHTML",
                            cls=combine_classes(btn, btn_colors.error, btn_sizes.xs)
                        )
                    )
                )
            )

        return Div(
            H2("Active Workers", cls=combine_classes(font_size._2xl, font_weight.bold, m.b(4))),
            Div(
                Table(
                    Thead(
                        Tr(
                            Th("PID"),
                            Th("Type"),
                            Th("Plugin"),
                            Th("Resource"),
                            Th("Device"),
                            Th("Status"),
                            Th("Actions")
                        )
                    ),
                    Tbody(*rows),
                    cls=combine_classes(table, table_modifiers.zebra, w.full)
                ),
                cls="overflow-x-auto"
            ),
            id="workers-list"
        )

    @rt("/workers/{pid}/remove")
    def post(pid: int):
        """Remove a worker."""
        resource_manager.unregister_worker(pid)
        print(f"[Demo] Removed worker PID {pid}")
        return workers_list()

    @rt("/validate")
    def post(plugin_id: str):
        """Validate resources for a plugin."""
        # Get plugin info
        plugin = plugin_registry.get_plugin(plugin_id)

        if not plugin:
            return Div(
                Div(
                    "❌ Plugin not found",
                    cls=combine_classes("alert", "alert-error")
                ),
                id="validation-result"
            )

        # Run validation
        result = validate_resources_for_job(
            resource_manager=resource_manager,
            plugin_registry=plugin_registry,
            get_plugin_resource_requirements=get_requirements_helper,
            compare_plugin_resources=compare_plugin_resources,
            get_plugin_resource_identifier=get_plugin_resource_identifier,
            plugin_id=plugin_id,
            verbose=True
        )

        # Determine alert style
        alert_cls = "alert-success" if result.can_proceed else "alert-error"
        if result.action == ValidationAction.RELOAD_PLUGIN:
            alert_cls = "alert-warning"

        # Build message
        details = []
        details.append(Div(Strong("Action: "), result.action.value))
        details.append(Div(Strong("Can proceed: "), str(result.can_proceed)))
        details.append(Div(Strong("Message: "), result.message))

        if result.current_worker:
            details.append(Div(Strong("Current worker: "), f"PID {result.current_worker.pid}"))

        if result.conflict:
            details.append(Div(Strong("Resource status: "), result.conflict.status.value))
            if result.conflict.app_pids:
                details.append(Div(Strong("App PIDs: "), ", ".join(map(str, result.conflict.app_pids))))
            if result.conflict.external_pids:
                details.append(Div(Strong("External PIDs: "), ", ".join(map(str, result.conflict.external_pids))))

        return Div(
            Div(
                H3(f"Validation Result: {result.action.value.upper()}",
                   cls=combine_classes(font_weight.bold, m.b(2))),
                *details,
                cls=combine_classes("alert", alert_cls)
            ),
            id="validation-result"
        )

    print("  ✓ FastHTML application configured")
    print("  ✓ Routes registered")

    print("\n" + "="*70)
    print("Demo App Ready!")
    print("="*70)
    print("\n📦 Library Components:")
    print("  • ResourceManager - Track workers and detect conflicts")
    print("  • Resource validation - Validate before job execution")
    print("  • Plugin utilities - Analyze plugin resource requirements")
    print("  • Configuration schemas - Monitoring and management config")

    print("\n🔌 Available Plugins:")
    for plugin in plugin_registry.plugins:
        is_local = is_local_model_plugin(plugin)
        plugin_type = "Local" if is_local else "API"
        print(f"  • {plugin.name} ({plugin_type})")

    print("\n📊 Configuration Schemas:")
    print(f"  • Resource Monitoring: {RESOURCE_MONITOR_SCHEMA['name']}")
    print(f"  • Resource Management: {RESOURCE_MANAGEMENT_SCHEMA['name']}")

    print("="*70 + "\n")

    return app


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    # Create the app
    app = main()

    def open_browser(url):
        print(f"🌐 Opening browser at {url}")
        webbrowser.open(url)

    port = 5040
    host = "0.0.0.0"
    display_host = 'localhost' if host in ['0.0.0.0', '127.0.0.1'] else host

    print(f"🚀 Server: http://{display_host}:{port}")
    print("\n📍 Available routes:")
    print(f"  http://{display_host}:{port}/          - Homepage with feature list")
    print(f"  http://{display_host}:{port}/manager   - Interactive resource manager")
    print("\n" + "="*70 + "\n")

    # Open browser after a short delay
    timer = threading.Timer(1.5, lambda: open_browser(f"http://localhost:{port}"))
    timer.daemon = True
    timer.start()

    # Start server
    uvicorn.run(app, host=host, port=port)
