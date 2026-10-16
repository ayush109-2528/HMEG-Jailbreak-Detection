import joblib
from ml.ensemble.hmeg import HMEG
from ml.ensemble.router import Router


class ModelStore:
    def __init__(self, config):
        self.bundle = None
        self.error = None
        self.engine = None
        target = config.model_dir / 'hmeg.joblib'
        if not target.is_file():
            return
        try:
            # Artifacts are installed by the researcher, never uploaded by visitors.
            bundle = joblib.load(target)
            if 'meaning' in bundle['experts']:
                bundle['experts']['meaning'].encoder()
            meta = bundle['metas'][bundle['preferred_meta']]
            self.engine = HMEG(bundle['experts'], meta, Router(config.low, config.high),
                               config.threshold, bundle['version'])
            self.bundle = bundle
        except Exception:
            self.error = 'Model artifacts or optional encoder unavailable. Check the trusted model bundle and installed dependencies.'

    def status(self) -> dict:
        return dict(models_loaded=self.engine is not None, model_error=self.error,
                    model_version=self.bundle['version'] if self.bundle else None,
                    meaning_available=bool(self.bundle and 'meaning' in self.bundle['experts']))
