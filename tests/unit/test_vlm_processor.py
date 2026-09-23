from app.multimodal.analysis import vlm_processor


def test_vlm_processor_registers_vlm_analyzer(monkeypatch):
    class FakeProvider:
        model = "fake-model"

    class FakeOCRProvider:
        pass

    class FakeAnalyzer:
        supported_type = "image"

        def __init__(self, provider, ocr_provider=None):
            self.provider = provider
            self.ocr_provider = ocr_provider

        def supports(self, content):
            return content.type == "image"

        def analyze(self, content):
            return None

    monkeypatch.setattr(
        vlm_processor,
        "GroqVLMProvider",
        lambda: FakeProvider(),
    )

    monkeypatch.setattr(
        vlm_processor,
        "PaddleOCRProvider",
        lambda: FakeOCRProvider(),
    )

    monkeypatch.setattr(
        vlm_processor,
        "VLMImageAnalyzer",
        FakeAnalyzer,
    )

    processor = (
        vlm_processor.create_vlm_analysis_processor()
    )

    assert len(processor.analyzers) == 3

    assert processor.analyzers[0].supported_type == "image"

    assert isinstance(
        processor.analyzers[0],
        FakeAnalyzer,
    )

    assert isinstance(
        processor.analyzers[0].ocr_provider,
        FakeOCRProvider,
    )

    assert processor.analyzers[1].supported_type == "table"
    assert processor.analyzers[2].supported_type == "equation"
def test_vlm_processor_wires_ocr_provider(monkeypatch):
    class FakeVLMProvider:
        model = "fake-model"

    class FakeOCRProvider:
        pass

    class FakeAnalyzer:
        supported_type = "image"

        def __init__(self, provider, ocr_provider=None):
            self.provider = provider
            self.ocr_provider = ocr_provider

        def supports(self, content):
            return content.type == "image"

        def analyze(self, content):
            return None

    fake_vlm = FakeVLMProvider()
    fake_ocr = FakeOCRProvider()

    monkeypatch.setattr(
        vlm_processor,
        "GroqVLMProvider",
        lambda: fake_vlm,
    )

    monkeypatch.setattr(
        vlm_processor,
        "PaddleOCRProvider",
        lambda: fake_ocr,
    )

    monkeypatch.setattr(
        vlm_processor,
        "VLMImageAnalyzer",
        FakeAnalyzer,
    )

    processor = vlm_processor.create_vlm_analysis_processor()

    image_analyzer = processor.analyzers[0]

    assert image_analyzer.provider is fake_vlm
    assert image_analyzer.ocr_provider is fake_ocr    


def test_vlm_processor_can_disable_ocr(monkeypatch):
    class FakeVLMProvider:
        model = "fake-model"

    class FakeAnalyzer:
        supported_type = "image"

        def __init__(self, provider, ocr_provider=None):
            self.provider = provider
            self.ocr_provider = ocr_provider

        def supports(self, content):
            return content.type == "image"

        def analyze(self, content):
            return None

    def fail_if_created():
        raise AssertionError("OCR provider should not be created")

    monkeypatch.setattr(
        vlm_processor,
        "GroqVLMProvider",
        lambda: FakeVLMProvider(),
    )
    monkeypatch.setattr(
        vlm_processor,
        "PaddleOCRProvider",
        fail_if_created,
    )
    monkeypatch.setattr(
        vlm_processor,
        "VLMImageAnalyzer",
        FakeAnalyzer,
    )

    processor = vlm_processor.create_vlm_analysis_processor(
        use_ocr=False
    )

    assert processor.analyzers[0].ocr_provider is None