import {
  HealthResponse,
  DatasetMetadata,
  DatasetSummary,
  DatasetPreviewResponse,
  ExperimentConfigRequest,
  ExperimentStatusResponse,
  ExperimentResultDetail,
  SHAPExplanationResponse,
  LeakageDiagnosticData,
  BenchmarksSummary,
} from "../types/api";

const BASE_URL = import.meta.env.VITE_API_URL || "";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorMsg = `HTTP Error ${res.status}: ${res.statusText}`;
    try {
      const errJson = await res.json();
      if (errJson.detail) {
        errorMsg = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (_) {
      // ignore
    }
    throw new Error(errorMsg);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE_URL}/api/health`);
    return handleResponse<HealthResponse>(res);
  },

  async getDatasets(): Promise<DatasetMetadata[]> {
    const res = await fetch(`${BASE_URL}/api/datasets`);
    return handleResponse<DatasetMetadata[]>(res);
  },

  async uploadDataset(file: File, targetColumn?: string): Promise<DatasetMetadata> {
    const formData = new FormData();
    formData.append("file", file);
    if (targetColumn) {
      formData.append("target_column", targetColumn);
    }
    const res = await fetch(`${BASE_URL}/api/datasets/upload`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<DatasetMetadata>(res);
  },

  async getDatasetSummary(datasetId: string, targetColumn?: string): Promise<DatasetSummary> {
    const url = targetColumn
      ? `${BASE_URL}/api/datasets/${datasetId}/summary?target_column=${encodeURIComponent(targetColumn)}`
      : `${BASE_URL}/api/datasets/${datasetId}/summary`;
    const res = await fetch(url);
    return handleResponse<DatasetSummary>(res);
  },

  async getDatasetPreview(datasetId: string, page = 1, pageSize = 25): Promise<DatasetPreviewResponse> {
    const res = await fetch(`${BASE_URL}/api/datasets/${datasetId}/preview?page=${page}&page_size=${pageSize}`);
    return handleResponse<DatasetPreviewResponse>(res);
  },

  async getExperiments(params?: {
    dataset_id?: string;
    oversampler?: string;
    classifier?: string;
    status?: string;
    is_benchmark?: boolean;
    limit?: number;
  }): Promise<ExperimentStatusResponse[]> {
    const searchParams = new URLSearchParams();
    if (params?.dataset_id) searchParams.append("dataset_id", params.dataset_id);
    if (params?.oversampler) searchParams.append("oversampler", params.oversampler);
    if (params?.classifier) searchParams.append("classifier", params.classifier);
    if (params?.status) searchParams.append("status", params.status);
    if (params?.is_benchmark !== undefined) searchParams.append("is_benchmark", String(params.is_benchmark));
    if (params?.limit) searchParams.append("limit", String(params.limit));

    const res = await fetch(`${BASE_URL}/api/experiments?${searchParams.toString()}`);
    return handleResponse<ExperimentStatusResponse[]>(res);
  },

  async createExperiment(config: ExperimentConfigRequest): Promise<ExperimentStatusResponse> {
    const res = await fetch(`${BASE_URL}/api/experiments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    return handleResponse<ExperimentStatusResponse>(res);
  },

  async getExperimentStatus(experimentId: string): Promise<ExperimentStatusResponse> {
    const res = await fetch(`${BASE_URL}/api/experiments/${experimentId}`);
    return handleResponse<ExperimentStatusResponse>(res);
  },

  async getExperimentResults(experimentId: string): Promise<ExperimentResultDetail> {
    const res = await fetch(`${BASE_URL}/api/experiments/${experimentId}/results`);
    return handleResponse<ExperimentResultDetail>(res);
  },

  async getExperimentExplanations(experimentId: string): Promise<SHAPExplanationResponse> {
    const res = await fetch(`${BASE_URL}/api/experiments/${experimentId}/explanations`);
    return handleResponse<SHAPExplanationResponse>(res);
  },

  async getLeakageDiagnostics(): Promise<LeakageDiagnosticData> {
    const res = await fetch(`${BASE_URL}/api/diagnostics/leakage`);
    return handleResponse<LeakageDiagnosticData>(res);
  },

  async getBenchmarksSummary(): Promise<BenchmarksSummary> {
    const res = await fetch(`${BASE_URL}/api/benchmarks/summary`);
    return handleResponse<BenchmarksSummary>(res);
  },

  getExportUrl(experimentId: string, format: "json" | "csv" | "markdown"): string {
    return `${BASE_URL}/api/experiments/${experimentId}/export?format=${format}`;
  },
};
