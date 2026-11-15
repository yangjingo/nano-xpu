import logging
import os
import threading

import uvicorn


class ManagedServer(uvicorn.Server):
    """Bound shutdown even if the engine thread stops returning from CUDA."""

    def __init__(self, app, config, shutdown_timeout=30):
        super().__init__(uvicorn.Config(
            app, host=config.host, port=config.port, workers=1,
            timeout_graceful_shutdown=shutdown_timeout,
        ))
        self.app = app
        self.shutdown_timeout = shutdown_timeout
        self.watchdog = None
        app.state.on_fatal = self.request_shutdown

    def force_shutdown(self):
        logging.getLogger(__name__).error("Engine shutdown timed out; forcing process exit")
        os._exit(1)

    def request_shutdown(self):
        self.app.state.service.ready = False
        self.should_exit = True
        if self.watchdog is None:
            self.watchdog = threading.Timer(self.shutdown_timeout, self.force_shutdown)
            self.watchdog.daemon = True
            self.watchdog.start()

    def handle_exit(self, sig, frame):
        super().handle_exit(sig, frame)
        self.request_shutdown()

    async def shutdown(self, sockets=None):
        self.request_shutdown()
        try:
            await super().shutdown(sockets=sockets)
        finally:
            self.watchdog.cancel()


def run_server(config, app):
    server = ManagedServer(app, config)
    server.run()
    if not server.started or app.state.service.failed:
        raise SystemExit(1)
