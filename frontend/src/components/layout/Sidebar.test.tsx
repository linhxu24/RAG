import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("renders every required control center tab", () => {
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>,
    );
    [
      "Chatbot",
      "Upload Documents",
      "Document Store",
      "Ingestion Monitor",
      "Retrieval Playground",
      "Evaluation Dashboard",
      "Observability",
      "Trace Explorer",
      "Asset Manager",
      "Data Tables",
      "Settings",
    ].forEach((label) => expect(screen.getByText(label)).toBeInTheDocument());
  });
});
