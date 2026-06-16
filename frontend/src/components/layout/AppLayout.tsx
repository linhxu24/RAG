import { Outlet, useLocation } from "react-router-dom";

import { PageErrorBoundary } from "../common/PageErrorBoundary";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";

export function AppLayout() {
  const location = useLocation();
  return (
    <div className="min-h-screen">
      <Sidebar />
      <Header />
      <main className="ml-[252px] min-h-screen pt-[72px]">
        <PageErrorBoundary resetKey={location.pathname}>
          <Outlet />
        </PageErrorBoundary>
      </main>
    </div>
  );
}
