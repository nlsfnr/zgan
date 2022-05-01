PYTHON=/usr/bin/env python3
PIP_FREEZE=.requirements.freeze.txt
PY_FILES=*.py dirk/ web/*.py
.PHONY: ci py-deps type-check lint

ci: $(PY_FILES) py-deps type-check lint

type-check: $(PY_FILES)
	$(PYTHON) -m mypy --install-types --strict $(PY_FILES)

lint: $(PY_FILES)
	$(PYTHON) -m flake8 $(PY_FILES)

py-deps: $(PIP_FREEZE)

$(PIP_FREEZE): requirements.txt
	$(PYTHON) -m pip install --upgrade pip && \
	$(PYTHON) -m pip install --upgrade -r requirements.txt && \
	$(PYTHON) -m pip freeze > $(PIP_FREEZE)
