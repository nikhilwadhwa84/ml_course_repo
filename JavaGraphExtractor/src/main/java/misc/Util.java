package misc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Util {
	
	

	public static List<File> collectFiles(File src) {
		return collectFiles(src, "java");
	}

	public static List<File> collectFiles(File src, String extension) {
		List<File> files = new ArrayList<>();
		for (File f : src.listFiles()) {
			if (f.isDirectory()) files.addAll(collectFiles(f, extension));
			else if (f.getName().endsWith(extension)) {
				files.add(f);
			}
		}
		return files;
	}
	
	public static List<String> readLines(File file) {
		try {
			CharsetDecoder dec = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.IGNORE);
			try (BufferedReader br = new BufferedReader(Channels.newReader(FileChannel.open(file.toPath()), dec, -1))) {
				List<String> lines = br.lines().collect(Collectors.toList());
				return lines;
			}
		} catch (IOException | UncheckedIOException e) {
			List<String> lines = new ArrayList<>();
			try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), StandardCharsets.UTF_8))) {
				String line;
				while ((line = br.readLine()) != null) {
					lines.add(line);
				}
			} catch (IOException e2) {
				e2.printStackTrace();
				return null;
			}
			return lines;
		}
	}

}
