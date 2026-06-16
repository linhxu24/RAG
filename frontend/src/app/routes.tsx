import { lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter, Navigate } from "react-router-dom";

import { AppLayout } from "../components/layout/AppLayout";
import { LoadingState } from "../components/common/States";

const ChatPage = lazy(() =>
  import("../components/chat/ChatPage").then((module) => ({ default: module.ChatPage })),
);
const UploadDocumentsPage = lazy(() =>
  import("../components/upload/UploadDocumentsPage").then((module) => ({
    default: module.UploadDocumentsPage,
  })),
);
const DocumentStorePage = lazy(() =>
  import("../components/documents/DocumentStorePage").then((module) => ({
    default: module.DocumentStorePage,
  })),
);
const IngestionMonitorPage = lazy(() =>
  import("../components/ingestion/IngestionMonitorPage").then((module) => ({
    default: module.IngestionMonitorPage,
  })),
);
const RetrievalPlaygroundPage = lazy(() =>
  import("../components/retrieval/RetrievalPlaygroundPage").then((module) => ({
    default: module.RetrievalPlaygroundPage,
  })),
);
const EvaluationDashboardPage = lazy(() =>
  import("../components/evaluation/EvaluationDashboardPage").then((module) => ({
    default: module.EvaluationDashboardPage,
  })),
);
const ObservabilityPage = lazy(() =>
  import("../components/observability/ObservabilityPage").then((module) => ({
    default: module.ObservabilityPage,
  })),
);
const TraceExplorerPage = lazy(() =>
  import("../components/traces/TraceExplorerPage").then((module) => ({
    default: module.TraceExplorerPage,
  })),
);
const AssetManagerPage = lazy(() =>
  import("../components/assets/AssetManagerPage").then((module) => ({
    default: module.AssetManagerPage,
  })),
);
const DataTablesPage = lazy(() =>
  import("../components/data/DataTablesPage").then((module) => ({
    default: module.DataTablesPage,
  })),
);
const SettingsPage = lazy(() =>
  import("../components/settings/SettingsPage").then((module) => ({
    default: module.SettingsPage,
  })),
);

function page(element: ReactNode) {
  return (
    <Suspense fallback={<div className="p-6"><LoadingState /></div>}>
      {element}
    </Suspense>
  );
}

export const router = createBrowserRouter([
  {
    element: <AppLayout />,
    children: [
      { index: true, element: <Navigate to="/chatbot" replace /> },
      { path: "chatbot", element: page(<ChatPage />) },
      { path: "upload", element: page(<UploadDocumentsPage />) },
      { path: "documents", element: page(<DocumentStorePage />) },
      { path: "ingestion", element: page(<IngestionMonitorPage />) },
      { path: "retrieval", element: page(<RetrievalPlaygroundPage />) },
      { path: "evaluation", element: page(<EvaluationDashboardPage />) },
      { path: "observability", element: page(<ObservabilityPage />) },
      { path: "traces", element: page(<TraceExplorerPage />) },
      { path: "assets", element: page(<AssetManagerPage />) },
      { path: "data", element: page(<DataTablesPage />) },
      { path: "settings", element: page(<SettingsPage />) },
      { path: "*", element: <Navigate to="/chatbot" replace /> },
    ],
  },
]);
