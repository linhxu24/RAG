import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";

import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("shows navigation tabs only after opening the sidebar menu", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <Sidebar />
      </MemoryRouter>,
    );

    expect(screen.queryByText("Upload Documents")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Mở danh sách trang" }));

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

    await user.keyboard("{Escape}");
    expect(screen.queryByText("Upload Documents")).not.toBeInTheDocument();
  });
});
