const API_BASE_URL = "http://127.0.0.1:8000";


async function fetchJSON(
  endpoint,
  options = {}
) {
  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    options
  );

  if (!response.ok) {
    let message =
      `API request failed: ${response.status}`;

    try {
      const errorData =
        await response.json();

      message =
        errorData.detail || message;

    } catch {
      // Keep default error message.
    }

    throw new Error(message);
  }

  return response.json();
}


export const api = {

  // ----------------------------------------------
  // Analytics
  // ----------------------------------------------

  getKpis: () =>
    fetchJSON("/api/kpis"),

  getMonthlyRevenue: () =>
    fetchJSON("/api/monthly-revenue"),

  getTopProducts: () =>
    fetchJSON("/api/top-products"),


  // ----------------------------------------------
  // Machine Learning
  // ----------------------------------------------

  getForecast: () =>
    fetchJSON("/api/forecast"),

  getAnomalies: () =>
    fetchJSON("/api/anomalies"),

  getCustomerSegments: () =>
    fetchJSON("/api/customer-segments"),

  getMlStatus: () =>
    fetchJSON("/api/ml-status"),


  // ----------------------------------------------
  // RAG
  // ----------------------------------------------

  getRagStatus: () =>
    fetchJSON("/api/rag/status"),


  uploadDocument: (file) => {

    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );

    return fetchJSON(
      "/api/rag/upload",
      {
        method: "POST",
        body: formData,
      }
    );
  },


  askRag: (
    question,
    top_k = 5
  ) =>
    fetchJSON(
      "/api/rag/ask",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          question,
          top_k,
        }),
      }
    ),


  // ----------------------------------------------
  // Combined AI Analyst
  // ----------------------------------------------

  askAnalyst: (
    question,
    top_k = 5
  ) =>
    fetchJSON(
      "/api/analyst/ask",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",
        },

        body: JSON.stringify({
          question,
          top_k,
        }),
      }
    ),
};