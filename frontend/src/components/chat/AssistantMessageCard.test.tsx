import { render, screen } from "@testing-library/react";

import type { ChatMessage } from "../../types";
import { AssistantMessageCard } from "./AssistantMessageCard";

describe("AssistantMessageCard", () => {
  it("renders a centered bounded assistant response", () => {
    const message: ChatMessage = {
      id: "1",
      role: "assistant",
      text: "Grounded answer",
      createdAt: new Date().toISOString(),
      response: {
        trace_id: "trace-1",
        answer: { text: "Grounded answer", assets: [], items: [], sources: [] },
      },
    };
    const { container } = render(<AssistantMessageCard message={message} />);
    expect(screen.getByText("Grounded answer")).toBeInTheDocument();
    expect(container.firstChild).toHaveClass("mx-auto", "max-w-[760px]");
  });
});
