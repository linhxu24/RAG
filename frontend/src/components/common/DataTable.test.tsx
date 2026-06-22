import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { DataTable, type Column } from "./DataTable";

interface Row {
  id: string;
  name: string;
  price: number;
}

const columns: Column<Row>[] = [
  {
    key: "name",
    label: "Name",
    render: (row) => row.name,
    sortValue: (row) => row.name,
    searchValue: (row) => row.name,
  },
  {
    key: "price",
    label: "Price",
    render: (row) => row.price,
    sortValue: (row) => row.price,
  },
];

describe("DataTable", () => {
  it("sorts and filters rows using column values", async () => {
    const user = userEvent.setup();
    render(
      <DataTable
        rows={[
          { id: "1", name: "Sản phẩm B", price: 20 },
          { id: "2", name: "Sản phẩm A", price: 10 },
        ]}
        columns={columns}
        rowKey={(row) => row.id}
      />,
    );

    await user.click(screen.getByRole("button", { name: /name/i }));
    const sortedRows = screen.getAllByRole("row");
    expect(sortedRows[1]).toHaveTextContent("Sản phẩm A");

    await user.type(screen.getByPlaceholderText("Tìm trong bảng..."), "B");
    expect(screen.getByText("Sản phẩm B")).toBeInTheDocument();
    expect(screen.queryByText("Sản phẩm A")).not.toBeInTheDocument();
  });
});
