"""Real isolated Stage B BGE test; no Qdrant, no legacy model/ingest code."""
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag3.gpu_model_service import GPUModelService, GPUServiceConfig


def main():
    import torch
    import psutil
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'
    before_free, total = torch.cuda.mem_get_info(0) if torch.cuda.is_available() else (0, 0)
    rss_before = psutil.Process().memory_info().rss
    svc = GPUModelService(GPUServiceConfig(default_deadline_seconds=300))
    observed = {'device': device, 'vram_total': total, 'vram_free_before': before_free,
                'rss_before': rss_before, 'cold_state': svc.snapshot()['state']}
    try:
        q = svc.submit_query('What is retrieval augmented generation?').result()
        docs = svc.submit_documents(['Document example %02d: retrieval augmented generation.' % n
                                     for n in range(17)]).result()
        again = svc.submit_query('Dense embedding query').result()
        # Reading device metadata is safe here: no extra inference is issued.
        # CUDA model must remain resident across both encode methods.
        param_devices = sorted({str(p.device) for p in svc._model.model.parameters()})
        free_after, _ = torch.cuda.mem_get_info(0)
        observed.update({'state': svc.snapshot()['state'], 'dimensions': len(q),
                         'query_norm': math.sqrt(sum(x*x for x in q)),
                         'document_count': len(docs),
                         'document_norms': [round(math.sqrt(sum(x*x for x in row)), 5) for row in docs],
                         'second_query_dimensions': len(again),
                         'model_parameter_devices': param_devices,
                         'metrics': svc.metrics(), 'rss_after': psutil.Process().memory_info().rss,
                         'vram_free_after': free_after,
                         'vram_used_delta': before_free-free_after})
        assert observed['cold_state'] == 'not_initialized'
        assert observed['state'] == 'ready'
        assert param_devices == ['cuda:0'], 'BGE instance did not remain GPU resident'
        assert observed['metrics']['model_load_count'] == 1
        assert observed['metrics']['document_batches'] == [8, 8, 1]
        assert len(q) == len(again) == len(docs[0])
        assert abs(observed['query_norm'] - 1) < .0001
        assert all(abs(v-1.0) < .0001 for v in observed['document_norms'])
        observed['result'] = 'PASS'
    except Exception as exc:
        observed['result'] = 'FAIL'
        observed['exception_type'] = type(exc).__name__
        observed['error_code'] = getattr(exc, 'code', 'UNKNOWN')
        observed['error_summary'] = str(exc)[:300]
        observed['service_status'] = svc.snapshot()
    finally:
        svc.close()
    print('B_REAL_GPU=' + json.dumps(observed, ensure_ascii=False, default=str))
    if observed['result'] != 'PASS':
        sys.exit(1)


if __name__ == '__main__':
    main()
