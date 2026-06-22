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

  it("renders product fields with user-facing labels and currency", () => {
    const message: ChatMessage = {
      id: "2",
      role: "assistant",
      text: "Danh sách sản phẩm",
      createdAt: new Date().toISOString(),
      response: {
        trace_id: "trace-2",
        answer: {
          text: "Danh sách sản phẩm",
          assets: [],
          sources: [],
          items: [
            {
              type: "product",
              id: "product-1",
              data: {
                name: "Bàn chải điện",
                category: "Chăm sóc răng",
                price: 2500000,
                currency: "VND",
                quantity: 3,
              },
            },
          ],
        },
      },
    };
    render(<AssistantMessageCard message={message} />);
    expect(screen.getByText("Tên")).toBeInTheDocument();
    expect(screen.getByText("2.500.000 ₫")).toBeInTheDocument();
    expect(screen.queryByText("product_id")).not.toBeInTheDocument();
  });
});
