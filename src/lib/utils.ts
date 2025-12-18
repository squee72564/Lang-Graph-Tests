import { Graph } from "@langchain/core/runnables/graph";
import fs from "fs";

export async function saveGraphToPng(graph: Graph, filePath: string) {
    const pngBlob = await graph.drawMermaidPng();
    const buffer = Buffer.from(await pngBlob.arrayBuffer());
    
    fs.writeFileSync(filePath, buffer);
}