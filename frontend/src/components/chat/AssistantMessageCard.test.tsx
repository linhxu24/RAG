import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

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

  it("renders contextual suggestions and returns the selected query", async () => {
    const user = userEvent.setup();
    const onSuggestion = vi.fn();
    const message: ChatMessage = {
      id: "3",
      role: "assistant",
      text: "Danh sách sản phẩm",
      createdAt: new Date().toISOString(),
      response: {
        trace_id: "trace-3",
        answer: {
          text: "Danh sách sản phẩm",
          assets: [],
          sources: [],
          items: [],
        },
        suggestions: [
          {
            suggestion_id: "sg-1",
            type: "next_question",
            label: "Chỉ xem sản phẩm còn hàng",
            query: "Sản phẩm nào còn hàng?",
            target_intent: "PRODUCT_LIST",
            reason_code: "refine_product_availability",
          },
        ],
      },
    };

    render(
      <AssistantMessageCard
        message={message}
        onSuggestion={onSuggestion}
      />,
    );
    await user.click(screen.getByRole("button", {
      name: "Chỉ xem sản phẩm còn hàng",
    }));

    expect(onSuggestion).toHaveBeenCalledWith(
      message.response?.suggestions?.[0],
    );
  });
});
