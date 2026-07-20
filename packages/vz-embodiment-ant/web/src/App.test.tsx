import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("truthful evidence surface", () => {
  it("starts in BLOCK and never presents replay as passed evidence", () => {
    render(<App />);
    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(
      screen.getByText(/尚无通过冻结门槛的正式 artifact/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "创建真实闭环" }),
    ).toBeInTheDocument();
  });
});
