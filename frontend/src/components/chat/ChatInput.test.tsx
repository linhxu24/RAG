import { fireEvent, render, screen } from "@testing-library/react";
import { vi } from "vitest";

import { ChatInput } from "./ChatInput";

describe("ChatInput", () => {
  it("sends on Enter and preserves Shift+Enter for newlines", () => {
    const send = vi.fn();
    render(
      <ChatInput value="hello" loading={false} onChange={() => undefined} onSend={send} />,
    );
    const textarea = screen.getByLabelText("Chat message");
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(send).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(send).toHaveBeenCalledTimes(1);
  });
});
