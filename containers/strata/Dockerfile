ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="K100AI Strata c4" \
      org.opencontainers.image.source="https://github.com/DocPang/strata-k100ai" \
      org.opencontainers.image.version="v0.1.0-rc.2"
COPY runtime/ /opt/k100ai/
COPY entrypoint.py /opt/k100ai/entrypoint.py
COPY profile.json /opt/k100ai/profile.json
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STRATA_GGUF_PY=/opt/k100ai/app/gguf-py \
    LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/dtk/lib:/opt/dtk/llvm/lib:/opt/dtk/.hyhal/lib
WORKDIR /opt/k100ai/app
# No apt/pip/network installation: the pinned base provides the tested dependencies.
RUN cd /opt/k100ai && sha256sum -c FILES.sha256 >/dev/null \
    && chmod 755 bin/strata \
    && cd /opt/k100ai/app \
    && python3 -c "import numpy,jinja2,regex,yaml,tqdm,requests,PIL,psutil; import serve.server"
ENTRYPOINT ["python3", "/opt/k100ai/entrypoint.py"]
CMD ["serve"]
